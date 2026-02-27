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

def main():
    log.info("ArbScanner starting up...")
    send_sms("ArbScanner live! Texts incoming when arbs found.")
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
def send_sms(message: str):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=YOUR_PHONE_NUMBER,
        )
        log.info(f"SMS sent: {message[:60]}...")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ─── KALSHI ──────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

def get_kalshi_token() -> str:
    """Log in with email+password and return a session token."""
    resp = requests.post(
        f"{KALSHI_BASE}/login",
        json={"email": KALSHI_EMAIL, "password": KALSHI_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("token", "")

_kalshi_token = ""
_kalshi_token_ts = 0

def fetch_kalshi_markets():
    """Returns list of {event, yes_price, no_price, market_id}"""
    global _kalshi_token, _kalshi_token_ts
    if not KALSHI_EMAIL or not KALSHI_PASSWORD:
        log.warning("No Kalshi credentials — skipping Kalshi")
        return []
    try:
        # Re-login if token is older than 20 minutes
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
            yes = m.get("yes_ask")   # cents (0–100)
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
POLY_BASE = "https://clob.polymarket.com"

def fetch_polymarket_markets():
    """Returns list of {event, yes_price, no_price, market_id}"""
    try:
        resp = requests.get(f"{POLY_BASE}/markets", timeout=10,
                            params={"active": True, "closed": False, "limit": 100})
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("data", data) if isinstance(data, dict) else data
        results = []
        for m in markets:
            tokens = m.get("tokens", [])
            yes_tok = next((t for t in tokens if t.get("outcome","").upper() == "YES"), None)
            no_tok  = next((t for t in tokens if t.get("outcome","").upper() == "NO"),  None)
            if not yes_tok or not no_tok:
                continue
            yes_p = float(yes_tok.get("price", 0))
            no_p  = float(no_tok.get("price", 0))
            if yes_p <= 0 or no_p <= 0:
                continue
            results.append({
                "source": "Polymarket",
                "event": m.get("question", "Unknown"),
                "market_id": m.get("condition_id", ""),
                "yes_price": yes_p,
                "no_price":  no_p,
                "url": f"https://polymarket.com/event/{m.get('slug','')}",
            })
        log.info(f"Polymarket: fetched {len(results)} markets")
        return results
    except Exception as e:
        log.error(f"Polymarket fetch error: {e}")
        return []

# ─── SPORTSBOOKS (via The Odds API) ──────────────────────────────────────────
SPORTS = ["americanfootball_nfl", "basketball_nba", "baseball_mlb",
          "icehockey_nhl", "soccer_epl"]
BOOKS  = ["draftkings", "fanduel", "betmgm", "pinnacle", "bovada"]

def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def decimal_to_prob(dec: float) -> float:
    return 1 / dec if dec > 0 else 0

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
                    "apiKey": ODDS_API_KEY,
                    "regions": "us",
                    "markets": "h2h",
                    "bookmakers": ",".join(BOOKS),
                    "oddsFormat": "american",
                },
                timeout=10,
            )
            resp.raise_for_status()
            games = resp.json()
            time.sleep(2)  # avoid rate limiting between sport requests

            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")

                # Collect ALL outcomes listed per book, preserving completeness
                bookmaker_data = {}   # {book: {team: decimal_odds}}
                all_outcome_names = set()

                for bm in game.get("bookmakers", []):
                    book_name = bm["key"]
                    for market in bm.get("markets", []):
                        if market["key"] != "h2h":
                            continue
                        book_outcomes = {}
                        for outcome in market.get("outcomes", []):
                            team  = outcome["name"]
                            price = american_to_decimal(outcome["price"])
                            book_outcomes[team] = price
                            all_outcome_names.add(team)
                        if book_outcomes:
                            bookmaker_data[book_name] = book_outcomes

                # Determine canonical team list for this game
                # For soccer this may include "Draw"; for others just 2 teams
                all_teams = sorted(all_outcome_names)

                if len(all_teams) < 2 or not bookmaker_data:
                    continue

                results.append({
                    "source":           "Sportsbooks",
                    "event":            f"{away} @ {home}",
                    "sport":            sport,
                    "market_id":        game.get("id", ""),
                    "commence_time":    game.get("commence_time", ""),
                    "_all_teams":       all_teams,
                    "_bookmaker_data":  bookmaker_data,
                })

        except Exception as e:
            log.error(f"Odds API error ({sport}): {e}")

    log.info(f"Sportsbooks: fetched {len(results)} games")
    return results

# ─── ARB DETECTION ────────────────────────────────────────────────────────────

def find_arbs_prediction_markets(markets_a: list, markets_b: list) -> list:
    """
    Compare YES/NO prices between two prediction market sources.
    Look for: buy YES on A + buy NO on B (or vice versa) where sum < 1.
    Uses fuzzy event title matching.
    """
    arbs = []
    for a in markets_a:
        for b in markets_b:
            # Simple keyword overlap check for same event
            words_a = set(a["event"].lower().split())
            words_b = set(b["event"].lower().split())
            overlap = words_a & words_b - {"the","a","an","of","in","to","vs","and","or","for","will"}
            if len(overlap) < 2:
                continue

            # Strategy 1: YES on A, NO on B
            sum1 = a["yes_price"] + b["no_price"]
            if sum1 < 1:
                profit = (1/sum1 - 1) * 100
                if profit > MIN_PROFIT_PCT:
                    arbs.append({
                        "event": a["event"],
                        "strategy": f"YES on {a['source']} @ {a['yes_price']:.3f}  +  NO on {b['source']} @ {b['no_price']:.3f}",
                        "profit_pct": profit,
                        "stake_a": (1/sum1) / a["yes_price"] * 100,
                        "stake_b": (1/sum1) / b["no_price"] * 100,
                        "url_a": a.get("url",""),
                        "url_b": b.get("url",""),
                    })

            # Strategy 2: NO on A, YES on B
            sum2 = a["no_price"] + b["yes_price"]
            if sum2 < 1:
                profit = (1/sum2 - 1) * 100
                if profit > MIN_PROFIT_PCT:
                    arbs.append({
                        "event": a["event"],
                        "strategy": f"NO on {a['source']} @ {a['no_price']:.3f}  +  YES on {b['source']} @ {b['yes_price']:.3f}",
                        "profit_pct": profit,
                        "stake_a": (1/sum2) / a["no_price"] * 100,
                        "stake_b": (1/sum2) / b["yes_price"] * 100,
                        "url_a": a.get("url",""),
                        "url_b": b.get("url",""),
                    })
    return arbs

def find_arbs_sportsbooks(games: list) -> list:
    """
    Proper arb detection for sportsbooks.

    Key rules:
    1. Every outcome in a game must come from a book that listed ALL outcomes
       for that game. If Book A only listed Home/Away but not Draw, we can't
       use it — mixing partial markets creates fake arbs.
    2. We pick the best decimal odds for each outcome across all valid books,
       but only from books where that book covered every outcome in the game.
    3. Cap realistic arbs at 15% — anything higher is a data error.
    """
    arbs = []

    for game in games:
        raw_bm_data = game.get("_bookmaker_data", {})  # {book: {team: decimal}}
        teams = game.get("_all_teams", [])

        if not raw_bm_data or not teams:
            continue

        # Only keep books that have odds for ALL teams in this game
        valid_books = {
            book: odds for book, odds in raw_bm_data.items()
            if all(team in odds for team in teams)
        }

        if not valid_books:
            continue

        # Best odds per team, only from valid (complete) books
        best = {}
        for team in teams:
            best_dec  = 0
            best_book = ""
            for book, odds in valid_books.items():
                if odds[team] > best_dec:
                    best_dec  = odds[team]
                    best_book = book
            if best_dec > 0:
                best[team] = {"decimal": best_dec, "book": best_book,
                              "implied_prob": 1 / best_dec}

        if len(best) != len(teams):
            continue  # couldn't find odds for every team

        implied_sum = sum(v["implied_prob"] for v in best.values())

        if implied_sum >= 1:
            continue  # no arb

        profit = (1 / implied_sum - 1) * 100

        # Sanity check: real arbs are almost never above 10-15%
        # Anything higher is bad data (mismatched markets, stale odds, etc.)
        if profit > 15:
            log.debug(f"Skipping suspicious arb ({profit:.1f}%) for {game['event']}")
            continue

        if profit <= MIN_PROFIT_PCT:
            continue

        # Build outcomes list for stake calculator
        payout_per_100 = 100 / implied_sum
        outcome_list = []
        for team in teams:
            stake = (best[team]["implied_prob"] / implied_sum) * 100
            outcome_list.append({
                "label": team,
                "book":  best[team]["book"],
                "price": best[team]["decimal"],
                "stake": stake,
            })

        strategy_parts = [
            f"{o['label']} on {o['book']} @ {o['price']:.2f}"
            for o in outcome_list
        ]

        arbs.append({
            "event":      game["event"],
            "strategy":   "  +  ".join(strategy_parts),
            "profit_pct": profit,
            "outcomes":   outcome_list,
            "implied_sum": implied_sum,
        })

    return arbs

# ─── STAKE CALCULATOR ─────────────────────────────────────────────────────────

def calc_stakes(outcomes: list, bankroll: float) -> dict:
    """
    outcomes: list of {"label": str, "price": float (decimal odds)}
    Returns dict with per-outcome stakes, guaranteed payout, and profit.
    """
    implied = [1 / o["price"] for o in outcomes]
    implied_sum = sum(implied)
    payout = bankroll / implied_sum
    profit = payout - bankroll

    stakes = [
        {
            "label": o["label"],
            "book":  o.get("book", ""),
            "price": o["price"],
            "stake": (implied[i] / implied_sum) * bankroll,
        }
        for i, o in enumerate(outcomes)
    ]
    return {"stakes": stakes, "payout": payout, "profit": profit}


BANKROLL_TIERS = [50, 100, 250, 500, 1000]

# ─── ALERT FORMATTING ─────────────────────────────────────────────────────────

def format_sms(arb: dict) -> str:
    """
    Concise SMS alert — stays under 320 chars (2 segments) so carriers don't block it.
    Full stake breakdown for $100 and $500 only. User can calculate others from the %.
    """
    outcomes = arb.get("outcomes", [])
    pct      = arb["profit_pct"]
    event    = arb["event"][:40]

    # Build bet instructions
    if outcomes:
        legs = " | ".join(f"{o['label']} on {o['book']} @{o['price']:.2f}" for o in outcomes)
        res100 = calc_stakes(outcomes, 100)
        res500 = calc_stakes(outcomes, 500)
        stakes100 = " + ".join(f"${s['stake']:.0f}" for s in res100["stakes"])
        stakes500 = " + ".join(f"${s['stake']:.0f}" for s in res500["stakes"])
        profit100 = res100["profit"]
        profit500 = res500["profit"]
    else:
        legs      = arb.get("strategy", "")[:80]
        profit100 = 100 * pct / 100
        profit500 = 500 * pct / 100
        stakes100 = f"${50 * pct / 100:.0f}+${50 * pct / 100:.0f}"
        stakes500 = "—"

    msg = (
        f"ARB +{pct:.2f}% | {event}\n"
        f"{legs}\n"
        f"$100 -> {stakes100} = +${profit100:.2f}\n"
        f"$500 -> {stakes500} = +${profit500:.2f}\n"
        f"{datetime.now().strftime('%H:%M')}"
    )

    # Hard cap at 300 chars to stay within 2 SMS segments
    return msg[:300]

# ─── DEDUP ────────────────────────────────────────────────────────────────────
# Store arb key → timestamp of last alert, so same arb doesn't re-alert for 4 hours
alerted_keys: dict = {}
ALERT_COOLDOWN_SEC = 4 * 3600  # 4 hours

def arb_key(arb: dict) -> str:
    # Key on event name + the books involved so lineup changes retrigger
    books = "|".join(sorted(o["book"] for o in arb.get("outcomes", [])))
    return f"{arb['event'][:40]}|{books}"

def is_new_arb(arb: dict) -> bool:
    key = arb_key(arb)
    last = alerted_keys.get(key, 0)
    if time.time() - last > ALERT_COOLDOWN_SEC:
        alerted_keys[key] = time.time()
        return True
    return False

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

def run_scan():
    log.info("═══ Starting scan ═══")

    kalshi_markets    = fetch_kalshi_markets()
    poly_markets      = fetch_polymarket_markets()
    sportsbook_games  = fetch_sportsbook_markets()

    all_arbs = []

    # Prediction market cross-platform arbs
    if kalshi_markets and poly_markets:
        arbs = find_arbs_prediction_markets(kalshi_markets, poly_markets)
        log.info(f"Kalshi ↔ Polymarket: {len(arbs)} arbs found")
        all_arbs.extend(arbs)

    # Sportsbook arbs
    sb_arbs = find_arbs_sportsbooks(sportsbook_games)
    log.info(f"Sportsbooks: {len(sb_arbs)} arbs found")
    all_arbs.extend(sb_arbs)

    # Sort by profit
    all_arbs.sort(key=lambda x: x["profit_pct"], reverse=True)

    # Alert on new ones only (respects 4hr cooldown)
    new_arbs = [a for a in all_arbs if is_new_arb(a)]

    if new_arbs:
        # Send top 3 individually with full breakdown
        log.info(f"📲 Sending {min(len(new_arbs), 3)} SMS alert(s) ({len(new_arbs)} total new arbs)")
        for arb in new_arbs[:3]:
            send_sms(format_sms(arb))
            time.sleep(1)  # small gap between sends
        if len(new_arbs) > 3:
            send_sms(f"⚡ +{len(new_arbs)-3} more arbs this scan (showing top 3). Best: +{new_arbs[0]['profit_pct']:.2f}%")
    else:
        log.info("No new arbs this scan.")

    log.info(f"═══ Scan complete. Next in {SCAN_INTERVAL_SEC}s ═══\n")

def main():
    log.info("ArbScanner Bot starting up...")
    send_sms("✅ ArbScanner is live! You'll receive SMS alerts when arbitrage opportunities are found.")
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan crashed: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
