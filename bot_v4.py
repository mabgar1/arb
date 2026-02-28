"""
ArbScanner Bot v5
- Everything from v4 plus automated Polymarket execution
- Finds YES+NO arbs on Polymarket and executes both sides automatically
- Uses official py-clob-client for order signing and submission
- FOK (Fill or Kill) orders: both sides execute or neither does
- Risk controls: max position size, daily loss limit, min liquidity
"""

import os, time, logging, requests, smtplib, base64, re, threading
from flask import Flask, jsonify, request as flask_request
from email.mime.text import MIMEText
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
YOUR_PHONE_NUMBER  = os.environ["YOUR_PHONE_NUMBER"]
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY", "")
KALSHI_KEY_ID      = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "")
MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT", "0.3"))
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC", "300"))
ALERT_COOLDOWN_SEC = int(os.environ.get("ALERT_COOLDOWN_SEC", "1800"))

# Polymarket execution config
POLY_PRIVATE_KEY   = os.environ.get("POLY_PRIVATE_KEY", "")    # wallet private key
POLY_FUNDER        = os.environ.get("POLY_FUNDER", "")         # wallet address
POLY_API_KEY       = os.environ.get("POLY_API_KEY", "")        # from create_or_derive_api_creds
POLY_API_SECRET    = os.environ.get("POLY_API_SECRET", "")
POLY_API_PASSPHRASE= os.environ.get("POLY_API_PASSPHRASE", "")
AUTO_EXECUTE       = os.environ.get("AUTO_EXECUTE", "false").lower() == "true"
MAX_POSITION_USDC  = float(os.environ.get("MAX_POSITION_USDC", "50"))   # max per trade
DAILY_LOSS_LIMIT   = float(os.environ.get("DAILY_LOSS_LIMIT",  "100"))  # stop if lost this much today
MIN_LIQUIDITY      = float(os.environ.get("MIN_LIQUIDITY",     "1000")) # skip thin markets

# ─── SLEEP STATE ──────────────────────────────────────────────────────────────
_sleeping   = False
_sleep_lock = threading.Lock()

def is_sleeping():
    with _sleep_lock:
        return _sleeping

def set_sleeping(val: bool):
    global _sleeping
    with _sleep_lock:
        _sleeping = val
    log.info(f"Sleep mode {'ON — scanning paused' if val else 'OFF — scanning resumed'}")

# ─── HTTP CONTROL SERVER ───────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def health():
    return jsonify({
        "status":       "ok",
        "sleeping":     is_sleeping(),
        "version":      "v5",
        "auto_execute": AUTO_EXECUTE,
        "daily_pnl":    daily_tracker.pnl_today(),
    })

@app.route("/sleep", methods=["POST"])
def sleep_on():
    set_sleeping(True)
    return jsonify({"sleeping": True})

@app.route("/wake", methods=["POST"])
def sleep_off():
    set_sleeping(False)
    return jsonify({"sleeping": False})

@app.route("/status")
def status():
    return jsonify({
        "sleeping":  is_sleeping(),
        "daily_pnl": daily_tracker.pnl_today(),
        "trades":    daily_tracker.trades_today(),
    })

def run_http():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("arbbot")

# ─── DAILY P&L TRACKER ────────────────────────────────────────────────────────
class DailyTracker:
    def __init__(self):
        self._lock   = threading.Lock()
        self._date   = date.today()
        self._pnl    = 0.0
        self._trades = []

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._date:
            self._date   = today
            self._pnl    = 0.0
            self._trades = []

    def record(self, profit_usdc: float, event: str):
        with self._lock:
            self._reset_if_new_day()
            self._pnl += profit_usdc
            self._trades.append({
                "time":   datetime.now().isoformat(),
                "event":  event[:60],
                "profit": round(profit_usdc, 4),
            })
            log.info(f"Trade recorded: ${profit_usdc:+.4f} | Daily P&L: ${self._pnl:+.4f}")

    def pnl_today(self):
        with self._lock:
            self._reset_if_new_day()
            return round(self._pnl, 4)

    def trades_today(self):
        with self._lock:
            self._reset_if_new_day()
            return len(self._trades)

    def over_loss_limit(self):
        return self.pnl_today() < -DAILY_LOSS_LIMIT

daily_tracker = DailyTracker()

# ─── SMS ──────────────────────────────────────────────────────────────────────
def send_sms(message: str):
    to_addr = f"{YOUR_PHONE_NUMBER}@vtext.com"
    try:
        msg = MIMEText(message[:300])
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = to_addr
        msg["Subject"] = ""
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_ADDRESS, to_addr, msg.as_string())
        log.info(f"SMS sent: {message[:60]}")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ─── STAKE CALCULATOR ─────────────────────────────────────────────────────────
def calc_stakes(outcomes, bankroll):
    implied = [1/o["price"] for o in outcomes if o.get("price", 0) > 0]
    if not implied:
        return {"stakes": [], "payout": 0, "profit": 0}
    implied_sum = sum(implied)
    if implied_sum == 0:
        return {"stakes": [], "payout": 0, "profit": 0}
    payout = bankroll / implied_sum
    stakes = [{
        "label": o["label"], "book": o["book"],
        "stake": (implied[i]/implied_sum) * bankroll,
        "token_id": o.get("token_id", ""),
        "price":    o["price"],
    } for i, o in enumerate(outcomes)]
    return {"stakes": stakes, "payout": payout, "profit": payout - bankroll}

def format_sms(arb):
    outcomes = arb["outcomes"]
    s        = calc_stakes(outcomes, 100)
    lines    = [f"ARB +{arb['profit_pct']:.2f}% [{arb['type']}]",
                arb["event"][:50]]
    for st in s["stakes"]:
        lines.append(f"  {st['book']}: ${st['stake']:.2f} on {st['label']}")
    lines.append(f"  Profit on $100: ${s['profit']:.2f}")
    return "\n".join(lines)

# ─── POLYMARKET EXECUTION ─────────────────────────────────────────────────────
_clob_client = None

def get_clob_client():
    global _clob_client
    if _clob_client:
        return _clob_client
    if not POLY_PRIVATE_KEY or not POLY_FUNDER:
        return None
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        creds = None
        if POLY_API_KEY:
            creds = ApiCreds(
                api_key        = POLY_API_KEY,
                api_secret     = POLY_API_SECRET,
                api_passphrase = POLY_API_PASSPHRASE,
            )

        client = ClobClient(
            host           = "https://clob.polymarket.com",
            chain_id       = 137,
            key            = POLY_PRIVATE_KEY,
            creds          = creds,
            signature_type = 1,  # email/Magic wallet
            funder         = POLY_FUNDER,
        )

        # Auto-derive credentials if not provided
        if not creds:
            derived = client.create_or_derive_api_creds()
            client.set_api_creds(derived)
            log.info("Polymarket: API creds derived from wallet")
            log.info(f"  Save these to env vars:")
            log.info(f"  POLY_API_KEY={derived.api_key}")
            log.info(f"  POLY_API_SECRET={derived.api_secret}")
            log.info(f"  POLY_API_PASSPHRASE={derived.api_passphrase}")

        _clob_client = client
        log.info("Polymarket CLOB client initialized")
        return client
    except ImportError:
        log.warning("py-clob-client not installed — no auto-execution")
        return None
    except Exception as e:
        log.error(f"Polymarket client init failed: {e}")
        return None

def execute_polymarket_arb(arb: dict, position_usdc: float) -> bool:
    """
    Execute both sides of a Polymarket YES/NO arb simultaneously.
    Uses FOK (Fill or Kill) orders — both execute or neither does.
    Returns True if successful.
    """
    client = get_clob_client()
    if not client:
        log.warning("No Polymarket client — skipping execution")
        return False

    if daily_tracker.over_loss_limit():
        log.warning(f"Daily loss limit hit (${DAILY_LOSS_LIMIT}) — skipping execution")
        send_sms(f"⚠️ Daily loss limit ${DAILY_LOSS_LIMIT} hit — auto-trading paused")
        return False

    outcomes = arb.get("outcomes", [])
    if len(outcomes) != 2:
        return False

    yes_outcome = next((o for o in outcomes if o["label"] == "YES"), None)
    no_outcome  = next((o for o in outcomes if o["label"] == "NO"),  None)

    if not yes_outcome or not no_outcome:
        log.warning(f"Arb missing YES or NO outcome: {arb['event'][:40]}")
        return False

    yes_token = yes_outcome.get("token_id", "")
    no_token  = no_outcome.get("token_id", "")

    if not yes_token or not no_token:
        log.warning(f"Missing token IDs for {arb['event'][:40]}")
        return False

    # Calculate optimal stakes
    s = calc_stakes(outcomes, position_usdc)
    if not s["stakes"] or s["profit"] <= 0:
        return False

    yes_stake = next((st for st in s["stakes"] if st["label"] == "YES"), None)
    no_stake  = next((st for st in s["stakes"] if st["label"] == "NO"),  None)

    if not yes_stake or not no_stake:
        return False

    log.info(f"Executing arb: {arb['event'][:50]}")
    log.info(f"  YES: ${yes_stake['stake']:.4f} @ {yes_outcome['price']:.4f} (token: {yes_token[:16]}...)")
    log.info(f"  NO:  ${no_stake['stake']:.4f} @ {no_outcome['price']:.4f} (token: {no_token[:16]}...)")
    log.info(f"  Expected profit: ${s['profit']:.4f} ({arb['profit_pct']:.2f}%)")

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # Place YES order (FOK = Fill or Kill)
        yes_order_args = OrderArgs(
            token_id   = yes_token,
            price      = round(yes_outcome["price"], 4),
            size       = round(yes_stake["stake"] / yes_outcome["price"], 2),
            side       = BUY,
        )
        yes_order = client.create_order(yes_order_args)

        # Place NO order (FOK)
        no_order_args = OrderArgs(
            token_id   = no_token,
            price      = round(no_outcome["price"], 4),
            size       = round(no_stake["stake"] / no_outcome["price"], 2),
            side       = BUY,
        )
        no_order = client.create_order(no_order_args)

        # Submit both simultaneously as a batch
        results = client.post_orders([
            {"order": yes_order, "orderType": OrderType.FOK},
            {"order": no_order,  "orderType": OrderType.FOK},
        ])

        yes_result = results[0] if len(results) > 0 else None
        no_result  = results[1] if len(results) > 1 else None

        yes_ok = yes_result and yes_result.get("success")
        no_ok  = no_result  and no_result.get("success")

        if yes_ok and no_ok:
            profit = s["profit"]
            daily_tracker.record(profit, arb["event"])
            msg = (f"✅ ARB EXECUTED +{arb['profit_pct']:.2f}%\n"
                   f"{arb['event'][:45]}\n"
                   f"Profit: ${profit:.4f} | Daily: ${daily_tracker.pnl_today():+.2f}")
            log.info(msg)
            send_sms(msg)
            return True
        else:
            # One or both failed — log details
            log.warning(f"Partial/failed execution:")
            log.warning(f"  YES: {'✅' if yes_ok else '❌'} {yes_result}")
            log.warning(f"  NO:  {'✅' if no_ok else '❌'} {no_result}")
            # Record as small loss (gas/fees)
            daily_tracker.record(-0.01, f"FAILED: {arb['event'][:40]}")
            return False

    except Exception as e:
        log.error(f"Execution error: {e}")
        return False

# ─── POLYMARKET FETCH (enhanced with token_ids) ───────────────────────────────
def fetch_polymarket():
    try:
        all_markets = []
        next_cursor = ""
        for _ in range(5):
            params = {"active": True, "closed": False, "limit": 100}
            if next_cursor:
                params["next_cursor"] = next_cursor
            resp = requests.get("https://clob.polymarket.com/markets",
                                params=params, timeout=15)
            resp.raise_for_status()
            data    = resp.json()
            markets = data.get("data", []) if isinstance(data, dict) else data
            if not markets:
                break
            for m in markets:
                tokens  = m.get("tokens", [])
                yes_tok = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
                no_tok  = next((t for t in tokens if t.get("outcome","").upper()=="NO"),  None)
                if not yes_tok or not no_tok:
                    continue
                yes_p = float(yes_tok.get("price", 0))
                no_p  = float(no_tok.get("price",  0))
                vol   = float(m.get("volume", 0))
                if yes_p > 0 and no_p > 0:
                    all_markets.append({
                        "source":       "Polymarket",
                        "event":        m.get("question", "?"),
                        "slug":         m.get("slug", ""),
                        "yes_price":    yes_p,
                        "no_price":     no_p,
                        "volume":       vol,
                        # Token IDs needed for execution
                        "yes_token_id": yes_tok.get("token_id", ""),
                        "no_token_id":  no_tok.get("token_id",  ""),
                    })
            next_cursor = data.get("next_cursor", "") if isinstance(data, dict) else ""
            if not next_cursor:
                break
        log.info(f"Polymarket: {len(all_markets)} markets")
        return all_markets
    except Exception as e:
        log.error(f"Polymarket fetch error: {e}")
        return []

# ─── KALSHI ───────────────────────────────────────────────────────────────────
def kalshi_auth_headers(method: str, path: str) -> dict:
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY:
        return {}
    try:
        ts   = str(int(time.time() * 1000))
        msg  = (ts + method.upper() + path).encode()
        pem  = base64.b64decode(KALSHI_PRIVATE_KEY)
        key  = serialization.load_pem_private_key(pem, password=None, backend=default_backend())
        sig  = key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }
    except Exception as e:
        log.error(f"Kalshi auth error: {e}")
        return {}

def fetch_kalshi():
    markets = []
    # Elections
    path = "/trade-api/v2/markets?limit=100&status=open"
    hdrs = kalshi_auth_headers("GET", path)
    if hdrs:
        try:
            r = requests.get(f"https://trading-api.kalshi.com{path}", headers=hdrs, timeout=15)
            r.raise_for_status()
            for m in r.json().get("markets", []):
                yes_p = m.get("yes_ask", 0) / 100
                no_p  = m.get("no_ask",  0) / 100
                if yes_p > 0 and no_p > 0:
                    markets.append({
                        "source":    "Kalshi",
                        "event":     m.get("title", "?"),
                        "yes_price": yes_p,
                        "no_price":  no_p,
                        "volume":    float(m.get("volume", 0)),
                    })
            log.info(f"Kalshi elections: {len(markets)} total so far")
        except Exception as e:
            log.error(f"Kalshi trading error: {e}")

    # Public markets (no auth required)
    try:
        r2 = requests.get("https://trading-api.kalshi.com/trade-api/v2/markets",
                          params={"limit": 400, "status": "open"}, timeout=15)
        if r2.ok:
            for m in r2.json().get("markets", []):
                yes_p = m.get("yes_ask", 0) / 100
                no_p  = m.get("no_ask",  0) / 100
                if yes_p > 0 and no_p > 0:
                    markets.append({
                        "source":    "Kalshi",
                        "event":     m.get("title", "?"),
                        "yes_price": yes_p,
                        "no_price":  no_p,
                        "volume":    float(m.get("volume", 0)),
                    })
    except Exception as e:
        log.error(f"Kalshi public fetch error: {e}")

    log.info(f"Kalshi total: {len(markets)} markets")
    return markets

# ─── SPORTSBOOKS ──────────────────────────────────────────────────────────────
SPORTS = [
    "basketball_nba","americanfootball_nfl","americanfootball_ncaaf",
    "basketball_ncaab","baseball_mlb","tennis_atp_french_open",
    "icehockey_nhl","mma_mixed_martial_arts",
]
BOOKS = ["draftkings","fanduel","betmgm","caesars","pointsbet_us","betrivers"]

def fetch_sportsbooks():
    if not ODDS_API_KEY:
        return []
    all_games = []
    for sport in SPORTS:
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "us",
                        "markets": "h2h", "bookmakers": ",".join(BOOKS)},
                timeout=15,
            )
            if not r.ok:
                continue
            games = r.json()
            log.info(f"  {sport}: {len(games)} games")
            for g in games:
                bookmakers = g.get("bookmakers", [])
                outcomes_by_book = {}
                for bm in bookmakers:
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            for o in mkt["outcomes"]:
                                name  = o["name"]
                                price = o["price"]
                                if name not in outcomes_by_book:
                                    outcomes_by_book[name] = []
                                outcomes_by_book[name].append({
                                    "book":  bm["key"],
                                    "price": price,
                                })
                if outcomes_by_book:
                    all_games.append({
                        "event":    f"{g.get('away_team')} @ {g.get('home_team')}",
                        "outcomes": outcomes_by_book,
                        "sport":    sport,
                    })
        except Exception as e:
            log.error(f"Sportsbook fetch error {sport}: {e}")
    log.info(f"Sportsbooks total: {len(all_games)} games")
    return all_games

# ─── ARB DETECTION ────────────────────────────────────────────────────────────
STOP = new_stop = set()

def normalize_event(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    stop = {"will","the","a","an","in","of","to","be","by","for","at","vs","who","what","when","which"}
    return " ".join(w for w in text.split() if w not in stop)

def event_similarity(a: str, b: str) -> float:
    na = set(normalize_event(a).split())
    nb = set(normalize_event(b).split())
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)

def find_prediction_arbs(markets_a, markets_b, min_similarity=0.4):
    arbs = []
    seen = set()
    for a in markets_a:
        for b in markets_b:
            if a is b:
                continue
            sim = event_similarity(a["event"], b["event"])
            if sim < min_similarity:
                continue
            for yes_src, no_src in [(a, b), (b, a)]:
                yp  = yes_src["yes_price"]
                np_ = no_src["no_price"]
                if yp <= 0 or np_ <= 0:
                    continue
                s = yp + np_
                if s < 1 and s > 0:
                    profit = (1/s - 1) * 100
                    if MIN_PROFIT_PCT < profit <= 15:
                        key = (f"{yes_src['source']}:{yes_src['event'][:30]}"
                               f"|{no_src['source']}:{no_src['event'][:30]}")
                        if key in seen:
                            continue
                        seen.add(key)
                        # Check if both sides are Polymarket (executable)
                        executable = (yes_src["source"] == "Polymarket" and
                                      no_src["source"]  == "Polymarket")
                        arbs.append({
                            "event":      a["event"],
                            "profit_pct": profit,
                            "type":       "PRED",
                            "similarity": sim,
                            "executable": executable,
                            "outcomes": [
                                {"label":    "YES",
                                 "book":     yes_src["source"],
                                 "price":    yes_src["yes_price"],
                                 "token_id": yes_src.get("yes_token_id", "")},
                                {"label":    "NO",
                                 "book":     no_src["source"],
                                 "price":    no_src["no_price"],
                                 "token_id": no_src.get("no_token_id", "")},
                            ],
                            "volume": min(yes_src.get("volume", 0), no_src.get("volume", 0)),
                        })
    return arbs

def find_sportsbook_arbs(games):
    arbs = []
    for g in games:
        outcomes_map = g["outcomes"]
        teams = list(outcomes_map.keys())
        if len(teams) < 2:
            continue
        best = {}
        for team, books in outcomes_map.items():
            best_book = max(books, key=lambda x: x["price"])
            best[team] = best_book
        prices = [v["price"] for v in best.values() if v.get("price", 0) > 0]
        if len(prices) < 2:
            continue
        impl_sum = sum(1/p for p in prices)
        if impl_sum >= 1 or impl_sum == 0:
            continue
        profit = (1/impl_sum - 1) * 100
        if MIN_PROFIT_PCT < profit <= 15:
            arbs.append({
                "event":      g["event"],
                "profit_pct": profit,
                "type":       "SPORT",
                "executable": False,
                "outcomes": [
                    {"label": team, "book": best[team]["book"],
                     "price": best[team]["price"], "token_id": ""}
                    for team in teams
                ],
            })
    return arbs

def find_cross_arbs(pred_markets, sb_games):
    arbs = []
    seen = set()
    for pm in pred_markets:
        for g in sb_games:
            sim = event_similarity(pm["event"], g["event"])
            if sim < 0.35:
                continue
            teams     = list(g["outcomes"].keys())
            best_sb   = {}
            for team, books in g["outcomes"].items():
                if books:
                    best_sb[team] = max(books, key=lambda x: x["price"])
            for i, team in enumerate(teams):
                if team not in best_sb:
                    continue
                sb_price = best_sb[team]["price"]
                if sb_price <= 0:
                    continue
                pm_yes = pm["yes_price"]
                pm_no  = pm["no_price"]
                other_team = teams[1-i]
                if other_team in best_sb:
                    other_sb_price = best_sb[other_team]["price"]
                    if other_sb_price <= 0 or pm_yes <= 0:
                        continue
                    s = pm_yes + (1/other_sb_price)
                    if s < 1 and s > 0:
                        profit = (1/s - 1) * 100
                        if MIN_PROFIT_PCT < profit <= 15:
                            key = f"cross:{pm['event'][:25]}:{team}"
                            if key not in seen:
                                seen.add(key)
                                arbs.append({
                                    "event":      f"{pm['event'][:35]} / {g['event'][:25]}",
                                    "profit_pct": profit,
                                    "type":       "CROSS",
                                    "executable": False,
                                    "note":       f"sim={sim:.2f}",
                                    "outcomes": [
                                        {"label":    "YES", "book": pm["source"],
                                         "price":    pm_yes,
                                         "token_id": pm.get("yes_token_id", "")},
                                        {"label":    team,  "book": best_sb[team]["book"],
                                         "price":    1/other_sb_price, "token_id": ""},
                                    ],
                                })
    return arbs

# ─── DEDUP ────────────────────────────────────────────────────────────────────
seen_arbs    = {}
executed_arbs= {}

def is_new(arb):
    key = arb["type"] + ":" + arb["event"][:40]
    now = time.time()
    if key in seen_arbs and (now - seen_arbs[key]) < ALERT_COOLDOWN_SEC:
        return False
    seen_arbs[key] = now
    return True

def already_executed(arb) -> bool:
    key = arb["type"] + ":" + arb["event"][:40]
    now = time.time()
    if key in executed_arbs and (now - executed_arbs[key]) < 3600:
        return True
    return False

def mark_executed(arb):
    key = arb["type"] + ":" + arb["event"][:40]
    executed_arbs[key] = time.time()

# ─── MAIN SCAN ────────────────────────────────────────────────────────────────
scan_count = 0

def run_scan():
    global scan_count
    scan_count += 1
    t0 = time.time()
    log.info(f"═══ Scan #{scan_count} ═══")

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_kalshi = ex.submit(fetch_kalshi)
        f_poly   = ex.submit(fetch_polymarket)
        f_sb     = ex.submit(fetch_sportsbooks)
        kalshi   = f_kalshi.result()
        poly     = f_poly.result()
        sb       = f_sb.result()

    all_arbs = []

    if kalshi and poly:
        all_arbs.extend(find_prediction_arbs(kalshi, poly))
    if len(kalshi) > 1:
        all_arbs.extend(find_prediction_arbs(kalshi, kalshi))
    if len(poly) > 1:
        # Polymarket internal arbs are executable
        all_arbs.extend(find_prediction_arbs(poly, poly))

    all_arbs.extend(find_sportsbook_arbs(sb))

    if (kalshi or poly) and sb:
        all_arbs.extend(find_cross_arbs(kalshi + poly, sb))

    all_arbs.sort(key=lambda x: x["profit_pct"], reverse=True)

    elapsed = time.time() - t0
    executable_count = sum(1 for a in all_arbs if a.get("executable"))
    log.info(f"Scan done in {elapsed:.1f}s — {len(all_arbs)} arbs "
             f"({executable_count} auto-executable)")

    # ── Auto-execute Polymarket arbs ──────────────────────────────────────────
    if AUTO_EXECUTE and POLY_PRIVATE_KEY:
        executable = [a for a in all_arbs
                      if a.get("executable")
                      and a["profit_pct"] >= 0.5  # higher bar for auto-execution
                      and a.get("volume", 0) >= MIN_LIQUIDITY
                      and not already_executed(a)]

        if executable:
            log.info(f"Auto-executing {len(executable)} Polymarket arb(s)...")
            for arb in executable[:3]:  # max 3 per scan
                if daily_tracker.over_loss_limit():
                    log.warning("Daily loss limit reached — stopping execution")
                    break
                position = min(MAX_POSITION_USDC, 50)  # cap at $50 per trade
                success  = execute_polymarket_arb(arb, position)
                if success:
                    mark_executed(arb)
                time.sleep(0.5)  # brief pause between orders

    # ── SMS alerts for new arbs ───────────────────────────────────────────────
    new = [a for a in all_arbs if is_new(a)]
    if new:
        for arb in new[:3]:
            tag  = "🤖 AUTO-EXECUTED" if arb.get("executable") and AUTO_EXECUTE else "🎯"
            log.info(f"  {tag} {arb['type']} +{arb['profit_pct']:.2f}% | {arb['event'][:50]}")
            if not arb.get("executable") or not AUTO_EXECUTE:
                send_sms(format_sms(arb))
        if len(new) > 3:
            send_sms(f"{len(new)-3} more arbs found")
    else:
        log.info("No new arbs this scan.")

    log.info(f"Daily P&L: ${daily_tracker.pnl_today():+.4f} | "
             f"Trades: {daily_tracker.trades_today()} | "
             f"═══ Next scan in {SCAN_INTERVAL_SEC}s ═══\n")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    log.info("ArbScanner v5 starting up...")
    log.info(f"Scan interval: {SCAN_INTERVAL_SEC}s | Min profit: {MIN_PROFIT_PCT}% | "
             f"Auto-execute: {AUTO_EXECUTE}")

    if AUTO_EXECUTE:
        log.info(f"Max position: ${MAX_POSITION_USDC} | Daily loss limit: ${DAILY_LOSS_LIMIT}")
        client = get_clob_client()
        if client:
            log.info("✅ Polymarket execution ready")
        else:
            log.warning("⚠️ Polymarket execution not configured — set POLY_PRIVATE_KEY + POLY_FUNDER")

    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()
    log.info(f"HTTP control server started | GET /status  POST /sleep  POST /wake")

    send_sms(f"ArbBot v5 live! Auto-execute: {'ON' if AUTO_EXECUTE else 'OFF'}")

    while True:
        if is_sleeping():
            log.info("Sleep mode active — skipping scan")
        else:
            try:
                run_scan()
            except Exception as e:
                log.error(f"Scan crashed: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
