"""
ArbScanner NY — New York Edition
- NY-licensed sportsbooks: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, PointsBet
- Kalshi prediction markets (CFTC-regulated, legal in NY)
- TheRundown API: delta polling (10s) + full snapshot (60s)
- Player props detection (requires TheRundown Starter tier)
- Flask HTTP: POST /sleep  POST /wake  GET /status
- SMS alerts via Gmail → Verizon gateway
"""

import os, time, logging, smtplib, threading, requests, base64
from datetime import datetime
from email.mime.text import MIMEText
from flask import Flask, jsonify
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
YOUR_PHONE_NUMBER  = os.environ["YOUR_PHONE_NUMBER"]
TR_API_KEY         = os.environ.get("TR_API_KEY", "")
KALSHI_KEY_ID      = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "")
MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT",     "0.5"))
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC",   "60"))
DELTA_INTERVAL_SEC = int(os.environ.get("DELTA_INTERVAL_SEC",  "10"))
ALERT_COOLDOWN_SEC = int(os.environ.get("ALERT_COOLDOWN_SEC",  "1800"))

# ─── NY SPORTSBOOKS (TheRundown affiliate IDs) ────────────────────────────────
NY_BOOKS = {
    3:  "DraftKings",
    24: "FanDuel",
    10: "BetMGM",
    22: "Caesars",
    35: "PointsBet",
    26: "BetRivers",
    68: "Resorts World",
    94: "Bally Bet",
}
NY_AFFILIATE_IDS = list(NY_BOOKS.keys())

# Game markets always available; props need Starter tier
TR_GAME_MARKETS  = [1, 2, 3]           # Moneyline, Spread, Total
TR_PROP_MARKETS  = [29, 35, 39, 28,    # Points, Rebounds, Assists, Pts+Reb+Ast
                    45, 52, 53, 54,    # Strikeouts, TDs, Rush Yds, Rec Yds
                    55, 68, 76, 77]    # Passing Yds, Goals, Saves, Hits

TR_SPORTS = {1:"NFL", 2:"NCAAF", 3:"MLB", 4:"NBA", 5:"NCAAB", 6:"NHL", 7:"WNBA", 9:"MMA"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("arbny")

# ─── SLEEP STATE ──────────────────────────────────────────────────────────────
_sleeping   = False
_sleep_lock = threading.Lock()

def is_sleeping():
    with _sleep_lock: return _sleeping

def set_sleeping(val: bool):
    global _sleeping
    with _sleep_lock: _sleeping = val
    state = "PAUSED" if val else "ACTIVE"
    log.info(f"{'─'*40}\n  Scanner {state}\n{'─'*40}")
    if val:   send_sms("🌙 ArbScanner paused")
    else:     send_sms("▶ ArbScanner resumed — scanning now")

# ─── FLASK ────────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def root():
    return jsonify({"service":"ArbScanner NY","sleeping":is_sleeping(),"status":"paused" if is_sleeping() else "scanning"})

@flask_app.route("/status")
def status(): return jsonify({"sleeping": is_sleeping()})

@flask_app.route("/sleep", methods=["POST"])
def sleep_route(): set_sleeping(True);  return jsonify({"sleeping": True})

@flask_app.route("/wake",  methods=["POST"])
def wake_route():  set_sleeping(False); return jsonify({"sleeping": False})

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), use_reloader=False)

# ─── SMS ──────────────────────────────────────────────────────────────────────
def send_sms(message: str):
    try:
        msg = MIMEText(message[:300])
        msg["From"] = GMAIL_ADDRESS
        msg["To"]   = f"{YOUR_PHONE_NUMBER}@vtext.com"
        msg["Subject"] = ""
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_ADDRESS, f"{YOUR_PHONE_NUMBER}@vtext.com", msg.as_string())
        log.info(f"SMS → {message[:60]}")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ─── KALSHI ───────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"
_kalshi_key = None

def _load_kalshi_key():
    global _kalshi_key
    if _kalshi_key: return _kalshi_key
    if not KALSHI_PRIVATE_KEY: return None
    try:
        pem = KALSHI_PRIVATE_KEY.replace("\\n", "\n").encode()
        _kalshi_key = serialization.load_pem_private_key(pem, password=None, backend=default_backend())
        return _kalshi_key
    except Exception as e:
        log.error(f"Kalshi key load failed: {e}")
        return None

def _kalshi_headers(method: str, path: str) -> dict:
    key = _load_kalshi_key()
    if not key: return {}
    ts  = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path.split("?")[0]).encode()
    sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KALSHI_KEY_ID, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "KALSHI-ACCESS-TIMESTAMP": ts}

def fetch_kalshi():
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY:
        log.info("Kalshi: no credentials — skipping")
        return []
    if not _load_kalshi_key():
        log.error("Kalshi: private key failed to load")
        return []
    all_markets = []
    cursor = ""
    for _ in range(10):  # up to 1000 markets
        path   = "/markets"
        params = {"limit": 100, "status": "open"}
        if cursor: params["cursor"] = cursor
        hdrs = _kalshi_headers("GET", path)
        try:
            r = requests.get(KALSHI_BASE + path, headers=hdrs, params=params, timeout=15)
            r.raise_for_status()
            data    = r.json()
            markets = data.get("markets", [])
            for m in markets:
                yes = m.get("yes_ask")
                no  = m.get("no_ask")
                if yes is None or no is None: continue
                yes_p = yes / 100
                no_p  = no  / 100
                if yes_p > 0 and no_p > 0:
                    all_markets.append({
                        "source":    "Kalshi",
                        "event":     m.get("title", "?"),
                        "ticker":    m.get("ticker", ""),
                        "yes_price": yes_p,
                        "no_price":  no_p,
                        "volume":    m.get("volume", 0),
                    })
            cursor = data.get("cursor", "")
            if not cursor: break
        except Exception as e:
            log.error(f"Kalshi fetch error: {e}")
            break
    log.info(f"Kalshi: {len(all_markets)} markets")
    return all_markets

# ─── THERUNDOWN ───────────────────────────────────────────────────────────────
TR_BASE    = "https://therundown.io/api/v2"
_delta_last_id = {}
_event_cache   = {}
_cache_lock    = threading.Lock()

def tr_get(path: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{TR_BASE}{path}", params={"key": TR_API_KEY, **(params or {})}, timeout=15)
        if r.status_code == 401: log.error("TheRundown: invalid API key"); return {}
        if r.status_code == 429: log.warning("Rate limit — backing off 30s"); time.sleep(30); return {}
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"TR API error {path}: {e}")
        return {}

def to_decimal(v) -> float:
    if not v or v == 0.0001: return 0
    p = float(v)
    if abs(p) >= 100: return (p/100)+1 if p > 0 else (100/abs(p))+1
    return p if p > 1 else 0

def parse_event(ev: dict, include_props: bool = False) -> dict | None:
    lines = ev.get("lines", {})
    if not lines: return None
    teams = ev.get("teams", [])
    if len(teams) < 2: return None
    home = teams[0].get("name", "Home")
    away = teams[1].get("name", "Away")
    event_name = f"{away} @ {home}"
    event_id   = ev.get("event_id", "")
    market_ids = TR_GAME_MARKETS + (TR_PROP_MARKETS if include_props else [])
    markets = {}

    for aff_id_str, line_data in lines.items():
        aff_id = int(aff_id_str)
        if aff_id not in NY_BOOKS: continue
        book = NY_BOOKS[aff_id]

        for mkt_id_str, mkt in line_data.items():
            mkt_id = int(mkt_id_str)
            if mkt_id not in market_ids: continue

            if mkt_id == 1:  # Moneyline
                label = "Moneyline"
                mh = to_decimal(mkt.get("moneyline_home"))
                ma = to_decimal(mkt.get("moneyline_away"))
                if mh > 1: markets.setdefault(label, {}).setdefault(home, {})[book] = mh
                if ma > 1: markets.setdefault(label, {}).setdefault(away, {})[book] = ma
                for p in mkt.get("participants", []):
                    pr = to_decimal(p.get("moneyline") or p.get("price"))
                    nm = p.get("name") or p.get("abbreviation") or ""
                    if pr > 1 and nm: markets.setdefault(label, {}).setdefault(nm, {})[book] = pr

            elif mkt_id == 2:  # Spread
                sh, sa = mkt.get("spread_home"), mkt.get("spread_away")
                ph = to_decimal(mkt.get("spread_home_line") or mkt.get("spread_price_home"))
                pa = to_decimal(mkt.get("spread_away_line") or mkt.get("spread_price_away"))
                if sh is not None and ph > 1:
                    s = ("+" if float(sh) > 0 else "") + str(sh)
                    markets.setdefault(f"Spread: {home} {s}", {}).setdefault("cover", {})[book] = ph
                if sa is not None and pa > 1:
                    s = ("+" if float(sa) > 0 else "") + str(sa)
                    markets.setdefault(f"Spread: {away} {s}", {}).setdefault("cover", {})[book] = pa
                for p in mkt.get("participants", []):
                    sp = p.get("spread") or p.get("handicap")
                    pr = to_decimal(p.get("spread_price") or p.get("price"))
                    nm = p.get("name") or p.get("abbreviation") or ""
                    if pr > 1 and nm and sp is not None:
                        s = ("+" if float(sp) > 0 else "") + str(sp)
                        markets.setdefault(f"Spread: {nm} {s}", {}).setdefault("cover", {})[book] = pr

            elif mkt_id == 3:  # Total
                total = mkt.get("total")
                po = to_decimal(mkt.get("over_price") or mkt.get("total_over_price"))
                pu = to_decimal(mkt.get("under_price") or mkt.get("total_under_price"))
                if total:
                    if po > 1: markets.setdefault(f"Total {total}", {}).setdefault("Over",  {})[book] = po
                    if pu > 1: markets.setdefault(f"Total {total}", {}).setdefault("Under", {})[book] = pu
                for p in mkt.get("participants", []):
                    t  = p.get("total") or p.get("point") or total
                    sd = p.get("name") or p.get("side") or ""
                    pr = to_decimal(p.get("total_price") or p.get("price"))
                    if pr > 1 and t and sd:
                        markets.setdefault(f"Total {t}", {}).setdefault(sd, {})[book] = pr

            elif mkt_id in TR_PROP_MARKETS and include_props:  # Player props
                for p in mkt.get("participants", []):
                    player = p.get("player_name") or p.get("name") or ""
                    if not player: continue
                    side  = p.get("side") or p.get("outcome") or ""
                    line  = p.get("line") or p.get("handicap") or ""
                    price = to_decimal(p.get("price") or p.get("moneyline"))
                    if price > 1 and side:
                        label = f"PROP: {player} {side} {line}".strip()
                        markets.setdefault(label, {}).setdefault("cover", {})[book] = price

    if not markets: return None
    return {"event_id": event_id, "event": event_name, "markets": markets}

def snapshot_sport(sport_id: int, include_props: bool = False):
    today    = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow = datetime.utcfromtimestamp(time.time() + 86400).strftime("%Y-%m-%d")
    mkt_ids  = TR_GAME_MARKETS + (TR_PROP_MARKETS if include_props else [])
    mkt_param = ",".join(str(m) for m in mkt_ids)
    aff_param = ",".join(str(a) for a in NY_AFFILIATE_IDS)
    count = 0
    for date_str in [today, tomorrow]:
        data = tr_get(f"/sports/{sport_id}/events/{date_str}", {"market_ids": mkt_param, "affiliate_ids": aff_param})
        for ev in data.get("events", []):
            parsed = parse_event(ev, include_props)
            if parsed:
                with _cache_lock: _event_cache[parsed["event_id"]] = parsed
                count += 1
        meta = data.get("meta", {})
        if meta.get("delta_last_id"): _delta_last_id[sport_id] = meta["delta_last_id"]
    log.info(f"  Snapshot {TR_SPORTS.get(sport_id, sport_id)}: {count} events")

def poll_delta(sport_id: int) -> int:
    last_id = _delta_last_id.get(sport_id)
    if not last_id: return 0
    aff_param = ",".join(str(a) for a in NY_AFFILIATE_IDS)
    mkt_param = ",".join(str(m) for m in TR_GAME_MARKETS)
    data = tr_get("/delta", {"last_id": last_id, "sport_id": sport_id, "market_ids": mkt_param, "affiliate_ids": aff_param})
    if not data: return 0
    if data.get("status") == 400:
        log.warning(f"Delta overflow sport {sport_id} — re-snapshotting")
        snapshot_sport(sport_id)
        return -1
    events  = data.get("events", [])
    meta    = data.get("meta", {})
    new_id  = meta.get("delta_last_id", "")
    if new_id: _delta_last_id[sport_id] = new_id
    updated = 0
    for ev in events:
        parsed = parse_event(ev)
        if parsed:
            with _cache_lock: _event_cache[parsed["event_id"]] = parsed
            updated += 1
    return updated

# ─── ARB DETECTION ────────────────────────────────────────────────────────────
def find_arbs_in_market(event: str, label: str, outcomes: dict) -> list:
    outcome_list = list(outcomes.items())
    if len(outcome_list) < 2: return []
    best = []
    for name, books in outcome_list:
        if not books: continue
        best_book, best_price = max(books.items(), key=lambda x: x[1])
        if best_price > 1: best.append({"name": name, "book": best_book, "price": best_price})
    if len(best) < 2: return []
    imp_sum = sum(1/o["price"] for o in best)
    if imp_sum >= 1 or imp_sum == 0: return []
    profit_pct = (1/imp_sum - 1) * 100
    if profit_pct < MIN_PROFIT_PCT: return []
    is_prop = label.startswith("PROP:")
    return [{"event": event, "label": label, "profit_pct": profit_pct,
             "legs": best, "implied_sum": imp_sum,
             "type": "prop" if is_prop else ("ml" if "Moneyline" in label else ("spread" if "Spread" in label else "total"))}]

def scan_sportsbook_arbs() -> list:
    arbs = []
    with _cache_lock: events = list(_event_cache.values())
    for ev in events:
        for label, outcomes in ev["markets"].items():
            arbs.extend(find_arbs_in_market(ev["event"], label, outcomes))
    return arbs

def find_kalshi_internal_arbs(kalshi_markets: list) -> list:
    """Find cases where YES+NO on same Kalshi market sum to <1 (different bid/ask windows)."""
    arbs = []
    for m in kalshi_markets:
        yes_p = m["yes_price"]
        no_p  = m["no_price"]
        if yes_p <= 0 or no_p <= 0: continue
        imp = yes_p + no_p  # these are already implied probabilities
        if imp < 1:
            profit_pct = (1/imp - 1) * 100 if imp > 0 else 0
            if profit_pct >= MIN_PROFIT_PCT:
                arbs.append({
                    "event":      m["event"],
                    "label":      "Kalshi YES+NO",
                    "profit_pct": profit_pct,
                    "type":       "kalshi",
                    "legs": [
                        {"name": "YES", "book": "Kalshi", "price": 1/yes_p},
                        {"name": "NO",  "book": "Kalshi", "price": 1/no_p},
                    ],
                    "implied_sum": imp,
                })
    return arbs

def all_arbs(kalshi_markets: list) -> list:
    arbs = scan_sportsbook_arbs()
    arbs.extend(find_kalshi_internal_arbs(kalshi_markets))
    return sorted(arbs, key=lambda x: x["profit_pct"], reverse=True)

# ─── STAKE CALC & SMS ─────────────────────────────────────────────────────────
def format_sms(arb: dict) -> str:
    bk = 500
    imp = arb.get("implied_sum") or sum(1/l["price"] for l in arb["legs"] if l["price"] > 1)
    if imp <= 0: return arb["event"][:100]
    profit = (bk/imp) - bk
    lines  = [f"ARB +{arb['profit_pct']:.2f}% [{arb['type'].upper()}]", arb["event"][:50]]
    for leg in arb["legs"]:
        stake = (1/leg["price"]) / imp * bk
        lines.append(f"  {leg['book']}: ${stake:.0f} on {leg['name']}")
    lines.append(f"  Profit on $500: ${profit:.2f}")
    return "\n".join(lines)

# ─── DEDUP ────────────────────────────────────────────────────────────────────
_seen = {}
def is_new(arb: dict) -> bool:
    key = f"{arb['event'][:40]}|{arb['label']}"
    now = time.time()
    if key in _seen and (now - _seen[key]) < ALERT_COOLDOWN_SEC: return False
    _seen[key] = now
    return True

# ─── LOOPS ────────────────────────────────────────────────────────────────────
_kalshi_cache = []
_kalshi_lock  = threading.Lock()

def kalshi_loop():
    """Refresh Kalshi every 30 seconds."""
    while True:
        if not is_sleeping():
            markets = fetch_kalshi()
            with _kalshi_lock: _kalshi_cache.clear(); _kalshi_cache.extend(markets)
        time.sleep(30)

def full_snapshot():
    log.info("Full snapshot of all sports...")
    include_props = bool(TR_API_KEY)  # props need Starter; try anyway
    for sport_id in TR_SPORTS:
        if not TR_API_KEY: break
        try: snapshot_sport(sport_id, include_props=include_props)
        except Exception as e: log.error(f"Snapshot {sport_id}: {e}")
        time.sleep(1)
    log.info(f"Snapshot done — {len(_event_cache)} events cached")

def delta_loop():
    while True:
        time.sleep(DELTA_INTERVAL_SEC)
        if is_sleeping() or not TR_API_KEY: continue
        updated = 0
        for sport_id in TR_SPORTS:
            try: updated += max(poll_delta(sport_id), 0)
            except Exception as e: log.error(f"Delta {sport_id}: {e}")
        if updated > 0:
            log.info(f"Delta: {updated} events updated")
            with _kalshi_lock: kalshi = list(_kalshi_cache)
            arbs = all_arbs(kalshi)
            new  = [a for a in arbs if is_new(a)]
            if new: alert(new)

def snapshot_loop():
    while True:
        time.sleep(SCAN_INTERVAL_SEC)
        if is_sleeping() or not TR_API_KEY: continue
        full_snapshot()
        with _kalshi_lock: kalshi = list(_kalshi_cache)
        arbs = all_arbs(kalshi)
        new  = [a for a in arbs if is_new(a)]
        log.info(f"Snapshot scan: {len(arbs)} arbs, {len(new)} new")
        if new: alert(new)

def alert(arbs: list):
    for arb in arbs[:3]:
        log.info(f"  🎯 +{arb['profit_pct']:.2f}% [{arb['type']}] {arb['event'][:45]}")
        send_sms(format_sms(arb))
        time.sleep(1)
    if len(arbs) > 3:
        send_sms(f"+{len(arbs)-3} more arbs. Best: +{arbs[0]['profit_pct']:.2f}%")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    log.info("═"*50)
    log.info("  ArbScanner NY")
    log.info(f"  Books: {', '.join(NY_BOOKS.values())}")
    log.info(f"  + Kalshi (CFTC-regulated, legal in NY)")
    log.info(f"  Delta: {DELTA_INTERVAL_SEC}s | Snapshot: {SCAN_INTERVAL_SEC}s | Min profit: {MIN_PROFIT_PCT}%")
    log.info("═"*50)

    threading.Thread(target=run_flask,      daemon=True).start()
    threading.Thread(target=kalshi_loop,    daemon=True).start()
    log.info("Control server up → POST /sleep | POST /wake | GET /status")

    if TR_API_KEY:
        full_snapshot()
        threading.Thread(target=delta_loop,    daemon=True).start()
        threading.Thread(target=snapshot_loop, daemon=True).start()
    else:
        log.warning("TR_API_KEY not set — add to Render env vars")

    send_sms("ArbScanner NY live! Books: DK/FD/BetMGM/Caesars+Kalshi")

    while True:
        time.sleep(60)
        if not is_sleeping():
            log.info(f"Heartbeat — {len(_event_cache)} events, scanning active")

if __name__ == "__main__":
    main()
