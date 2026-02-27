"""
ArbScanner Bot
Scans Kalshi, Polymarket, and sportsbooks for arbitrage opportunities.
Sends SMS alerts FREE via Verizon email-to-SMS gateway (no Twilio needed).
"""

import os
import time
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ─── CONFIG (set these as environment variables in Render) ────────────────────
# Gmail credentials — use a Gmail App Password (NOT your real password)
# Generate one at: myaccount.google.com/apppasswords
GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]        # yourname@gmail.com
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]   # 16-char app password

# Your Verizon phone number — digits only, no dashes or +1
# e.g. if your number is (516) 690-0512, enter: 5166900512
YOUR_PHONE_NUMBER  = os.environ["YOUR_PHONE_NUMBER"]

KALSHI_EMAIL       = os.environ.get("KALSHI_EMAIL", "")
KALSHI_PASSWORD    = os.environ.get("KALSHI_PASSWORD", "")
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY", "")

MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT", "0"))
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC", "600"))

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("arbbot")

# ─── FREE SMS via Verizon email gateway ───────────────────────────────────────
def send_sms(message: str):
    to_addr = f"{YOUR_PHONE_NUMBER}@vtext.com"
    try:
        msg = MIMEText(message[:300])
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = to_addr
        msg["Subject"] = ""
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_addr, msg.as_string())
        log.info(f"SMS sent: {message[:60]}...")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ─── KALSHI ───────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"
_kalshi_token = ""
_kalshi_token_ts = 0

def get_kalshi_token() -> str:
    resp = requests.post(
        f"{KALSHI_BASE}/login",
        json={"email": KALSHI_EMAIL, "password": KALSHI_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("token", "")

def fetch_kalshi_markets():
    global _kalshi_token, _kalshi_token_ts
    if not KALSHI_EMAIL or not KALSHI_PASSWORD:
        log.warning("No Kalshi credentials — skipping Kalshi")
        return []
    try:
        if time.time() - _kalshi_token_ts > 1200:
            _kalshi_token = get_kalshi_token()
            _kalshi_token_ts = time.time()
            log.info("Kalshi: obtained fresh session token")
        headers = {"Authorization": f"Bearer {_kalshi_token}"}
        resp = requests.get(f"{KALSHI_BASE}/markets", headers=headers, timeout=10,
                            params={"limit": 100, "status": "open"})
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        results = []
        for m in markets:
            yes = m.get("yes_ask")
            no  = m.get("no_ask")
            if yes is None or no is None:
                continue
            results.append({
                "source": "Kalshi",
                "event": m.get("title", "Unknown"),
                "market_id": m.get("ticker", ""),
                "yes_price": yes / 100,
                "no_price":  no  / 100,
                "url": f"https://kalshi.com/markets/{m.get('ticker','')}",
            })
        log.info(f"Kalshi: fetched {len(results)} markets")
        return results
    except Exception as e:
        log.error(f"Kalshi fetch error: {e}")
        return []

# ─── POLYMARKET ───────────────────────────────────────────────────────────────
def fetch_polymarket_markets():
    try:
        resp = requests.get("https://clob.polymarket.com/markets", timeout=10,
                            params={"active": True, "closed": False, "limit": 100})
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("data", data) if isinstance(data, dict) else data
        results = []
        for m in markets:
            tokens  = m.get("tokens", [])
            yes_tok = next((t for t in tokens if t.get("outcome","").upper() == "YES"), None)
            no_tok  = next((t for t in tokens if t.get("outcome","").upper() == "NO"),  None)
            if not yes_tok or not no_tok:
                continue
            yes_p = float(yes_tok.get("price", 0))
            no_p  = float(no_tok.get("price",  0))
            if yes_p <= 0 or no_p <= 0:
                continue
            results.append({
                "source":    "Polymarket",
                "event":     m.get("question", "Unknown"),
                "market_id": m.get("condition_id", ""),
                "yes_price": yes_p,
                "no_price":  no_p,
                "url":       f"https://polymarket.com/event/{m.get('slug','')}",
            })
        log.info(f"Polymarket: fetched {len(results)} markets")
        return results
    except Exception as e:
        log.error(f"Polymarket fetch error: {e}")
        return []

# ─── SPORTSBOOKS ──────────────────────────────────────────────────────────────
SPORTS = ["americanfootball_nfl", "basketball_nba", "baseball_mlb", "icehockey_nhl"]
BOOKS  = ["draftkings", "fanduel", "betmgm", "pinnacle", "bovada"]

def american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

def fetch_sportsbook_markets():
    if not ODDS_API_KEY:
        log.warning("No Odds API key — skipping sportsbooks")
        return []
    results = []
    for sport in SPORTS:
        try:
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                params={
                    "apiKey":      ODDS_API_KEY,
                    "regions":     "us",
                    "markets":     "h2h",
                    "bookmakers":  ",".join(BOOKS),
                    "oddsFormat":  "american",
                },
                timeout=10,
            )
            resp.raise_for_status()
            time.sleep(2)
            for game in resp.json():
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                bookmaker_data   = {}
                all_outcome_names = set()
                for bm in game.get("bookmakers", []):
                    book_outcomes = {}
                    for market in bm.get("markets", []):
                        if market["key"] != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            team  = outcome["name"]
                            price = american_to_decimal(outcome["price"])
                            book_outcomes[team] = price
                            all_outcome_names.add(team)
                    if book_outcomes:
                        bookmaker_data[bm["key"]] = book_outcomes
                all_teams = sorted(all_outcome_names)
                if len(all_teams) >= 2 and bookmaker_data:
                    results.append({
                        "source":          "Sportsbooks",
                        "event":           f"{away} @ {home}",
                        "sport":           sport,
                        "_all_teams":      all_teams,
                        "_bookmaker_data": bookmaker_data,
                    })
        except Exception as e:
            log.error(f"Odds API error ({sport}): {e}")
    log.info(f"Sportsbooks: fetched {len(results)} games")
    return results

# ─── ARB DETECTION ────────────────────────────────────────────────────────────
def find_arbs_prediction_markets(markets_a, markets_b):
    arbs = []
    for a in markets_a:
        for b in markets_b:
            words_a = set(a["event"].lower().split())
            words_b = set(b["event"].lower().split())
            stop    = {"the","a","an","of","in","to","vs","and","or","for","will"}
            if len((words_a & words_b) - stop) < 2:
                continue
            for yes_src, no_src in [(a, b), (b, a)]:
                s = yes_src["yes_price"] + no_src["no_price"]
                if s < 1:
                    profit = (1/s - 1) * 100
                    if profit > MIN_PROFIT_PCT:
                        arbs.append({
                            "event":      a["event"],
                            "profit_pct": profit,
                            "outcomes": [
                                {"label":"YES","book":yes_src["source"],"price":yes_src["yes_price"]},
                                {"label":"NO", "book":no_src["source"], "price":no_src["no_price"]},
                            ],
                        })
    return arbs

def find_arbs_sportsbooks(games):
    arbs = []
    for game in games:
        raw   = game.get("_bookmaker_data", {})
        teams = game.get("_all_teams", [])
        if not raw or not teams:
            continue
        # Only books that have ALL teams
        valid = {b: o for b, o in raw.items() if all(t in o for t in teams)}
        if not valid:
            continue
        best = {}
        for team in teams:
            best_dec, best_book = 0, ""
            for book, odds in valid.items():
                if odds[team] > best_dec:
                    best_dec, best_book = odds[team], book
            if best_dec > 0:
                best[team] = {"decimal": best_dec, "book": best_book}
        if len(best) != len(teams):
            continue
        implied_sum = sum(1/v["decimal"] for v in best.values())
        if implied_sum >= 1:
            continue
        profit = (1/implied_sum - 1) * 100
        if profit > 15 or profit <= MIN_PROFIT_PCT:
            continue
        outcomes = [
            {"label": t, "book": best[t]["book"], "price": best[t]["decimal"]}
            for t in teams
        ]
        arbs.append({"event": game["event"], "profit_pct": profit, "outcomes": outcomes})
    return arbs

# ─── STAKE CALCULATOR ─────────────────────────────────────────────────────────
def calc_stakes(outcomes, bankroll):
    implied     = [1/o["price"] for o in outcomes]
    implied_sum = sum(implied)
    payout      = bankroll / implied_sum
    stakes      = [{"label":o["label"],"book":o["book"],
                    "stake":(implied[i]/implied_sum)*bankroll}
                   for i, o in enumerate(outcomes)]
    return {"stakes": stakes, "payout": payout, "profit": payout - bankroll}

# ─── SMS FORMATTING ───────────────────────────────────────────────────────────
def format_sms(arb):
    outcomes = arb.get("outcomes", [])
    pct      = arb["profit_pct"]
    event    = arb["event"][:38]
    legs     = " | ".join(f"{o['label']}@{o['book']}({o['price']:.2f})" for o in outcomes)
    r100     = calc_stakes(outcomes, 100)
    r500     = calc_stakes(outcomes, 500)
    s100     = "+".join(f"${s['stake']:.0f}" for s in r100["stakes"])
    s500     = "+".join(f"${s['stake']:.0f}" for s in r500["stakes"])
    return (
        f"ARB +{pct:.2f}% | {event}\n"
        f"{legs}\n"
        f"$100: {s100} = +${r100['profit']:.2f}\n"
        f"$500: {s500} = +${r500['profit']:.2f}\n"
        f"{datetime.now().strftime('%H:%M')}"
    )[:300]

# ─── DEDUP ────────────────────────────────────────────────────────────────────
alerted_keys: dict = {}
ALERT_COOLDOWN_SEC = 4 * 3600

def is_new_arb(arb):
    books = "|".join(sorted(o["book"] for o in arb.get("outcomes", [])))
    key   = f"{arb['event'][:40]}|{books}"
    if time.time() - alerted_keys.get(key, 0) > ALERT_COOLDOWN_SEC:
        alerted_keys[key] = time.time()
        return True
    return False

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def run_scan():
    log.info("═══ Starting scan ═══")
    kalshi   = fetch_kalshi_markets()
    poly     = fetch_polymarket_markets()
    sb_games = fetch_sportsbook_markets()

    all_arbs = []
    if kalshi and poly:
        pm_arbs = find_arbs_prediction_markets(kalshi, poly)
        log.info(f"Kalshi↔Polymarket: {len(pm_arbs)} arbs")
        all_arbs.extend(pm_arbs)
    sb_arbs = find_arbs_sportsbooks(sb_games)
    log.info(f"Sportsbooks: {len(sb_arbs)} arbs")
    all_arbs.extend(sb_arbs)
    all_arbs.sort(key=lambda x: x["profit_pct"], reverse=True)

    new_arbs = [a for a in all_arbs if is_new_arb(a)]
    if new_arbs:
        log.info(f"📲 Sending {min(3,len(new_arbs))} alerts ({len(new_arbs)} total)")
        for arb in new_arbs[:3]:
            send_sms(format_sms(arb))
            time.sleep(1)
        if len(new_arbs) > 3:
            send_sms(f"ARB: +{len(new_arbs)-3} more this scan. Best: +{new_arbs[0]['profit_pct']:.2f}%")
    else:
        log.info("No new arbs this scan.")
    log.info(f"═══ Done. Next in {SCAN_INTERVAL_SEC}s ═══\n")

def send_test_alert():
    """Sends a sample alert so you can see exactly what real arb texts look like."""
    fake_arb = {
        "event": "Lakers vs Warriors (EXAMPLE)",
        "profit_pct": 1.87,
        "outcomes": [
            {"label": "Lakers",   "book": "draftkings", "price": 2.10},
            {"label": "Warriors", "book": "pinnacle",   "price": 2.05},
        ],
    }
    send_sms("** EXAMPLE ARB ALERT **\n" + format_sms(fake_arb))
    log.info("Test alert sent.")

def main():
    log.info("ArbScanner starting up...")
    send_sms("ArbScanner live! Texts incoming when arbs found.")
    send_test_alert()
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
