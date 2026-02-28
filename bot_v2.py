"""
ArbScanner Bot v3
- Scans every 5 minutes (was 30)
- 10 sportsbooks (was 5)
- Smarter Kalshi↔Polymarket matching using normalized team/event names
- Cross-platform: Kalshi/Polymarket vs sportsbooks
- Concurrent fetching for speed
- Cooldown reduced to 30min (was 4hr) so re-alerts if arb persists
"""

import os, time, logging, requests, smtplib, base64, re
from email.mime.text import MIMEText
from datetime import datetime
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
MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT", "0.3"))  # ignore tiny arbs
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC", "300"))  # 5 min
ALERT_COOLDOWN_SEC = int(os.environ.get("ALERT_COOLDOWN_SEC", "1800")) # 30 min re-alert

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("arbbot")

# ─── SMS ──────────────────────────────────────────────────────────────────────
def send_sms(message: str):
    to_addr = f"{YOUR_PHONE_NUMBER}@vtext.com"
    try:
        msg = MIMEText(message[:300])
        msg["From"] = GMAIL_ADDRESS
        msg["To"]   = to_addr
        msg["Subject"] = ""
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_ADDRESS, to_addr, msg.as_string())
        log.info(f"SMS sent: {message[:60]}...")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ─── STAKE CALCULATOR ─────────────────────────────────────────────────────────
def calc_stakes(outcomes, bankroll):
    implied     = [1/o["price"] for o in outcomes]
    implied_sum = sum(implied)
    payout      = bankroll / implied_sum
    stakes      = [{"label":o["label"],"book":o["book"],
                    "stake":(implied[i]/implied_sum)*bankroll}
                   for i,o in enumerate(outcomes)]
    return {"stakes":stakes, "payout":payout, "profit":payout-bankroll}

def format_sms(arb):
    outcomes = arb["outcomes"]
    pct      = arb["profit_pct"]
    legs     = " | ".join(f"{o['label']}@{o['book']}({o['price']:.2f})" for o in outcomes)
    r100     = calc_stakes(outcomes, 100)
    r500     = calc_stakes(outcomes, 500)
    s100     = "+".join(f"${s['stake']:.0f}" for s in r100["stakes"])
    s500     = "+".join(f"${s['stake']:.0f}" for s in r500["stakes"])
    tag      = arb.get("type","").upper()
    return (
        f"[{tag}] ARB +{pct:.2f}%\n"
        f"{arb['event'][:40]}\n"
        f"{legs}\n"
        f"$100:{s100}=+${r100['profit']:.2f} "
        f"$500:{s500}=+${r500['profit']:.2f}\n"
        f"{datetime.now().strftime('%H:%M')}"
    )[:300]

# ─── KALSHI ───────────────────────────────────────────────────────────────────
KALSHI_BASE_TRADING   = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_BASE_ELECTIONS = "https://api.elections.kalshi.com/trade-api/v2"
_kalshi_key = None

def _load_kalshi_key():
    if not KALSHI_PRIVATE_KEY:
        return None
    try:
        pem = KALSHI_PRIVATE_KEY.replace("\\n", "\n").encode()
        return serialization.load_pem_private_key(pem, password=None,
                                                   backend=default_backend())
    except Exception as e:
        log.error(f"Kalshi key load failed: {e}")
        return None

def _sign_pss(private_key, text: str) -> str:
    message = text.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode("utf-8")

def _kalshi_headers(method: str, path: str) -> dict:
    global _kalshi_key
    if _kalshi_key is None:
        _kalshi_key = _load_kalshi_key()
    if _kalshi_key is None:
        return {}
    ts_ms = str(int(time.time() * 1000))
    sig   = _sign_pss(_kalshi_key, ts_ms + method.upper() + path.split("?")[0])
    return {"KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms}

def _fetch_kalshi_page(base_url: str, label: str, cursor: str = "") -> tuple:
    path = "/trade-api/v2/markets"
    params = {"limit": 100, "status": "open"}
    if cursor:
        params["cursor"] = cursor
    hdrs = _kalshi_headers("GET", path)
    resp = requests.get(base_url + "/markets", headers=hdrs,
                        params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])
    next_cursor = data.get("cursor", "")
    results = []
    for m in markets:
        yes = m.get("yes_ask")
        no  = m.get("no_ask")
        if yes is None or no is None:
            continue
        results.append({
            "source":    f"Kalshi",
            "event":     m.get("title", "?"),
            "ticker":    m.get("ticker", ""),
            "yes_price": yes / 100,
            "no_price":  no  / 100,
            "volume":    m.get("volume", 0),
        })
    return results, next_cursor

def fetch_kalshi():
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY:
        log.warning("No Kalshi credentials — skipping")
        return []
    if _load_kalshi_key() is None:
        log.error("Kalshi: private key failed to load")
        return []
    all_results = []
    for base, label in [(KALSHI_BASE_ELECTIONS, "elections"), (KALSHI_BASE_TRADING, "trading")]:
        try:
            cursor = ""
            pages  = 0
            while pages < 5:  # fetch up to 500 markets per endpoint
                results, cursor = _fetch_kalshi_page(base, label, cursor)
                all_results.extend(results)
                pages += 1
                if not cursor:
                    break
            log.info(f"Kalshi {label}: {sum(1 for r in all_results if True)} total so far")
        except Exception as e:
            log.error(f"Kalshi {label} error: {e}")
    log.info(f"Kalshi total: {len(all_results)} markets")
    return all_results

# ─── POLYMARKET ───────────────────────────────────────────────────────────────
def fetch_polymarket():
    try:
        all_markets = []
        # Fetch multiple pages
        next_cursor = ""
        for _ in range(5):  # up to 500 markets
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
                if yes_p > 0 and no_p > 0:
                    all_markets.append({
                        "source":    "Polymarket",
                        "event":     m.get("question", "?"),
                        "slug":      m.get("slug", ""),
                        "yes_price": yes_p,
                        "no_price":  no_p,
                        "volume":    float(m.get("volume", 0)),
                    })
            next_cursor = data.get("next_cursor", "") if isinstance(data, dict) else ""
            if not next_cursor:
                break
        log.info(f"Polymarket: {len(all_markets)} markets")
        return all_markets
    except Exception as e:
        log.error(f"Polymarket error: {e}")
        return []

# ─── SMART EVENT MATCHING ─────────────────────────────────────────────────────
# Normalize team/event names for better cross-platform matching
TEAM_ALIASES = {
    # NBA
    "lakers": ["los angeles lakers", "la lakers", "lal"],
    "celtics": ["boston celtics", "bos"],
    "warriors": ["golden state warriors", "gsw", "golden state"],
    "nets": ["brooklyn nets", "bkn"],
    "knicks": ["new york knicks", "nyk"],
    "heat": ["miami heat", "mia"],
    "bulls": ["chicago bulls", "chi"],
    "bucks": ["milwaukee bucks", "mil"],
    "76ers": ["philadelphia 76ers", "phi", "sixers"],
    "suns": ["phoenix suns", "phx"],
    # NFL
    "chiefs": ["kansas city chiefs", "kc"],
    "eagles": ["philadelphia eagles", "phi"],
    "cowboys": ["dallas cowboys", "dal"],
    "patriots": ["new england patriots", "ne"],
    "packers": ["green bay packers", "gb"],
    "niners": ["san francisco 49ers", "sf", "49ers"],
    "ravens": ["baltimore ravens", "bal"],
    "bills": ["buffalo bills", "buf"],
    # MLB
    "yankees": ["new york yankees", "nyy"],
    "dodgers": ["los angeles dodgers", "lad"],
    "red sox": ["boston red sox", "bos"],
    "cubs": ["chicago cubs", "chc"],
    "astros": ["houston astros", "hou"],
}

def normalize_event(text: str) -> str:
    """Lowercase, remove punctuation, expand aliases."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Expand known aliases
    for canonical, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            t = t.replace(alias, canonical)
    return t

def event_similarity(a: str, b: str) -> float:
    """Return 0-1 similarity score between two event strings."""
    na = normalize_event(a)
    nb = normalize_event(b)
    stop = {"the","a","an","of","in","to","vs","and","or","for","will","by","at",
            "who","what","when","win","beat","game","match","series","over","under"}
    wa = set(na.split()) - stop
    wb = set(nb.split()) - stop
    if not wa or not wb:
        return 0
    overlap = wa & wb
    return len(overlap) / max(len(wa), len(wb))

# ─── PREDICTION MARKET CROSS-ARBS ─────────────────────────────────────────────
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
            for yes_src, no_src in [(a,b),(b,a)]:
                s = yes_src["yes_price"] + no_src["no_price"]
                if s < 1:
                    profit = (1/s - 1) * 100
                    if MIN_PROFIT_PCT < profit <= 15:
                        key = f"{yes_src['source']}:{yes_src['event'][:30]}|{no_src['source']}:{no_src['event'][:30]}"
                        if key in seen:
                            continue
                        seen.add(key)
                        arbs.append({
                            "event":      a["event"],
                            "profit_pct": profit,
                            "type":       "PRED",
                            "similarity": sim,
                            "outcomes": [
                                {"label":"YES","book":yes_src["source"],
                                 "price":yes_src["yes_price"]},
                                {"label":"NO", "book":no_src["source"],
                                 "price":no_src["no_price"]},
                            ]
                        })
    return arbs

# ─── SPORTSBOOKS ──────────────────────────────────────────────────────────────
# More sports + more books vs v2
SPORTS = [
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_nba",
    "basketball_ncaab",
    "baseball_mlb",
    "icehockey_nhl",
    "mma_mixed_martial_arts",
    "tennis_atp_french_open",
]

BOOKS = [
    "draftkings", "fanduel", "betmgm", "pinnacle", "bovada",
    "caesars", "pointsbet_us", "bet365", "betrivers", "williamhill_us",
]

def to_dec(odds):
    return (odds/100)+1 if odds>0 else (100/abs(odds))+1

def _fetch_sport(sport: str) -> list:
    """Fetch one sport — called concurrently."""
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
        params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h",
                "bookmakers": ",".join(BOOKS), "oddsFormat": "american"},
        timeout=15
    )
    r.raise_for_status()
    results = []
    for game in r.json():
        bm_data, all_teams = {}, set()
        for bm in game.get("bookmakers", []):
            book_odds = {}
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for oc in mkt.get("outcomes", []):
                    book_odds[oc["name"]] = to_dec(oc["price"])
                    all_teams.add(oc["name"])
            if book_odds:
                bm_data[bm["key"]] = book_odds
        all_teams = sorted(all_teams)
        if len(all_teams) >= 2 and bm_data:
            results.append({
                "event":   f"{game.get('away_team')} @ {game.get('home_team')}",
                "sport":   sport,
                "time":    game.get("commence_time", ""),
                "_teams":  all_teams,
                "_bm":     bm_data,
            })
    return results

def fetch_sportsbooks():
    if not ODDS_API_KEY:
        return []
    results = []
    # Fetch all sports concurrently
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_sport, s): s for s in SPORTS}
        for future in as_completed(futures):
            sport = futures[future]
            try:
                games = future.result()
                results.extend(games)
                log.info(f"  {sport}: {len(games)} games")
            except Exception as e:
                log.error(f"Odds API ({sport}): {e}")
    log.info(f"Sportsbooks total: {len(results)} games")
    return results

def find_sportsbook_arbs(games):
    arbs = []
    for g in games:
        teams = g["_teams"]
        valid = {b:o for b,o in g["_bm"].items() if all(t in o for t in teams)}
        if not valid:
            continue
        best = {}
        for t in teams:
            bd, bb = max(((o[t], b) for b, o in valid.items()), key=lambda x: x[0])
            best[t] = {"price": bd, "book": bb}
        impl_sum = sum(1/v["price"] for v in best.values())
        if impl_sum >= 1:
            continue
        profit = (1/impl_sum - 1) * 100
        if profit > 15 or profit <= MIN_PROFIT_PCT:
            continue
        outcomes = [{"label":t,"book":best[t]["book"],"price":best[t]["price"]}
                    for t in teams]
        arbs.append({
            "event":      g["event"],
            "profit_pct": profit,
            "type":       "SPORT",
            "sport":      g.get("sport",""),
            "outcomes":   outcomes,
        })
    return arbs

# ─── CROSS-PLATFORM: Prediction markets vs sportsbooks ────────────────────────
def find_cross_arbs(pred_markets, sb_games):
    """
    Compare Kalshi/Polymarket implied prob vs sportsbook decimal odds.
    If Kalshi says 60% but best sportsbook pays 2.10 (47.6%), that's mispricing.
    We alert: buy YES on Kalshi + bet opposite on sportsbook (or vice versa).
    """
    arbs = []
    seen = set()

    for pm in pred_markets:
        for g in sb_games:
            # Match event
            sim = event_similarity(pm["event"], g["event"])
            if sim < 0.45:
                continue

            teams = g["_teams"]
            valid = {b:o for b,o in g["_bm"].items() if all(t in o for t in teams)}
            if not valid or len(teams) != 2:
                continue

            # Get best sportsbook odds for each side
            best_sb = {}
            for t in teams:
                bd, bb = max(((o[t], b) for b, o in valid.items()), key=lambda x: x[0])
                best_sb[t] = {"price": bd, "book": bb}

            # YES on pred market vs NO-equivalent on sportsbook
            # Pred market YES price = probability of YES
            # Sportsbook: team A wins at decimal D => implied prob = 1/D
            for i, team in enumerate(teams):
                sb_price = best_sb[team]["price"]
                sb_book  = best_sb[team]["book"]
                sb_prob  = 1 / sb_price  # implied prob

                # If pred market YES is cheaper than sportsbook implied prob
                pm_yes   = pm["yes_price"]
                pm_no    = pm["no_price"]

                # Strategy: buy YES on pred market + back opposite team on sportsbook
                # This works if pm_yes + (1 - sb_prob_team_A) < 1
                # i.e., pm_yes + sb_prob_team_B < 1
                other_team = teams[1-i]
                if other_team in best_sb:
                    other_sb_price = best_sb[other_team]["price"]
                    # Combined: pm_yes_price + 1/other_sb_price < 1?
                    s = pm_yes + (1/other_sb_price)
                    if s < 1:
                        profit = (1/s - 1) * 100
                        if MIN_PROFIT_PCT < profit <= 15:
                            key = f"cross:{pm['event'][:25]}:{team}"
                            if key not in seen:
                                seen.add(key)
                                arbs.append({
                                    "event":      f"{pm['event'][:35]} / {g['event'][:25]}",
                                    "profit_pct": profit,
                                    "type":       "CROSS",
                                    "note":       f"sim={sim:.2f}",
                                    "outcomes": [
                                        {"label":"YES","book":pm["source"],
                                         "price":pm_yes},
                                        {"label":other_team,"book":sb_book,
                                         "price":other_sb_price},
                                    ]
                                })
    return arbs

# ─── DEDUP ────────────────────────────────────────────────────────────────────
alerted: dict = {}

def is_new(arb):
    key = arb["type"] + ":" + arb["event"][:40] + "|" + \
          "|".join(sorted(o["book"] for o in arb["outcomes"]))
    if time.time() - alerted.get(key, 0) > ALERT_COOLDOWN_SEC:
        alerted[key] = time.time()
        return True
    return False

# ─── MAIN SCAN ────────────────────────────────────────────────────────────────
scan_count = 0

def run_scan():
    global scan_count
    scan_count += 1
    t0 = time.time()
    log.info(f"═══ Scan #{scan_count} ═══")

    # Fetch all sources concurrently
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_kalshi = ex.submit(fetch_kalshi)
        f_poly   = ex.submit(fetch_polymarket)
        f_sb     = ex.submit(fetch_sportsbooks)
        kalshi = f_kalshi.result()
        poly   = f_poly.result()
        sb     = f_sb.result()

    all_arbs = []

    # Kalshi ↔ Polymarket
    if kalshi and poly:
        arbs = find_prediction_arbs(kalshi, poly)
        log.info(f"Kalshi↔Polymarket: {len(arbs)} arbs")
        all_arbs.extend(arbs)

    # Kalshi internal (different markets on same platform)
    if len(kalshi) > 1:
        arbs = find_prediction_arbs(kalshi, kalshi)
        log.info(f"Kalshi internal: {len(arbs)} arbs")
        all_arbs.extend(arbs)

    # Polymarket internal
    if len(poly) > 1:
        arbs = find_prediction_arbs(poly, poly)
        log.info(f"Polymarket internal: {len(arbs)} arbs")
        all_arbs.extend(arbs)

    # Sportsbook pure arbs
    sb_arbs = find_sportsbook_arbs(sb)
    log.info(f"Sportsbook pure arbs: {len(sb_arbs)}")
    all_arbs.extend(sb_arbs)

    # Cross-platform: prediction markets vs sportsbooks
    if (kalshi or poly) and sb:
        pred = kalshi + poly
        cross = find_cross_arbs(pred, sb)
        log.info(f"Cross-platform arbs: {len(cross)}")
        all_arbs.extend(cross)

    # Sort by profit, deduplicate, alert
    all_arbs.sort(key=lambda x: x["profit_pct"], reverse=True)
    new = [a for a in all_arbs if is_new(a)]

    elapsed = time.time() - t0
    log.info(f"Scan done in {elapsed:.1f}s — {len(all_arbs)} arbs total, {len(new)} new")

    if new:
        for arb in new[:3]:
            log.info(f"  🎯 {arb['type']} +{arb['profit_pct']:.2f}% | {arb['event'][:50]}")
            send_sms(format_sms(arb))
            time.sleep(1)
        if len(new) > 3:
            send_sms(f"{len(new)-3} more arbs. Best: {new[0]['type']} +{new[0]['profit_pct']:.2f}%")
    else:
        log.info("No new arbs this scan.")

    log.info(f"═══ Next scan in {SCAN_INTERVAL_SEC}s ═══\n")

def main():
    log.info("ArbScanner v3 starting up...")
    log.info(f"Scan interval: {SCAN_INTERVAL_SEC}s | Min profit: {MIN_PROFIT_PCT}% | Cooldown: {ALERT_COOLDOWN_SEC}s")
    send_sms(f"ArbBot v3 live! Scanning every {SCAN_INTERVAL_SEC//60}min across Kalshi+Polymarket+{len(SPORTS)} sports+{len(BOOKS)} books")
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan crashed: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
