"""
ArbScanner Bot
Scans Kalshi, Polymarket, and sportsbooks for arbitrage opportunities.
Sends SMS alerts via Twilio when profit > threshold.
"""

import os
import time
import logging
import requests
from datetime import datetime
from twilio.rest import Client

# ─── CONFIG (set these as environment variables) ─────────────────────────────
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN  = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM_NUMBER = os.environ["TWILIO_FROM_NUMBER"]   # e.g. +15551234567
YOUR_PHONE_NUMBER  = os.environ["YOUR_PHONE_NUMBER"]    # e.g. +15559876543

KALSHI_EMAIL       = os.environ.get("KALSHI_EMAIL", "")
KALSHI_PASSWORD    = os.environ.get("KALSHI_PASSWORD", "")
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY", "")         # from the-odds-api.com

MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT", "0"))  # 0 = any positive EV
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC", "60"))

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("arbbot")

# ─── SMS ─────────────────────────────────────────────────────────────────────
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
                bookmaker_data = {}
                for bm in game.get("bookmakers", []):
                    book_name = bm["key"]
                    for market in bm.get("markets", []):
                        if market["key"] != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            team = outcome["name"]
                            price = american_to_decimal(outcome["price"])
                            if book_name not in bookmaker_data:
                                bookmaker_data[book_name] = {}
                            bookmaker_data[book_name][team] = price

                # Find best odds for each team across all books
                teams = [home, away]
                best = {}
                for team in teams:
                    best_dec = 0
                    best_book = ""
                    for book, odds in bookmaker_data.items():
                        if team in odds and odds[team] > best_dec:
                            best_dec = odds[team]
                            best_book = book
                    if best_dec > 0:
                        best[team] = {"decimal": best_dec, "book": best_book,
                                      "implied_prob": decimal_to_prob(best_dec)}

                if len(best) == 2:
                    t1, t2 = teams
                    results.append({
                        "source": "Sportsbooks",
                        "event": f"{away} @ {home}",
                        "sport": sport,
                        "outcomes": best,
                        "market_id": game.get("id", ""),
                        "commence_time": game.get("commence_time", ""),
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
    arbs = []
    for game in games:
        outcomes = game["outcomes"]
        teams = list(outcomes.keys())
        if len(teams) < 2:
            continue
        implied_sum = sum(o["implied_prob"] for o in outcomes.values())
        if implied_sum < 1:
            profit = (1/implied_sum - 1) * 100
            if profit > MIN_PROFIT_PCT:
                strategy_parts = [
                    f"{team} on {outcomes[team]['book']} @ {outcomes[team]['decimal']:.2f}"
                    for team in teams
                ]
                arbs.append({
                    "event": game["event"],
                    "strategy": "  +  ".join(strategy_parts),
                    "profit_pct": profit,
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
    Build an SMS with a full stake breakdown table for each bankroll tier.
    Also shows the implied margin (guaranteed % profit).
    """
    outcomes = arb.get("outcomes", [])

    lines = [
        f"⚡ ARB — {arb['profit_pct']:.2f}% guaranteed",
        f"📋 {arb['event'][:55]}",
        "",
    ]

    # Show where to bet each side
    if outcomes:
        for o in outcomes:
            lines.append(f"  {o['label']:12s} → {o['book']:12s} @ {o['price']:.3f}")
    else:
        lines.append(f"  {arb.get('strategy','')[:80]}")

    lines.append("")
    lines.append("💰 STAKE BREAKDOWN")
    lines.append(f"{'Bankroll':>9} | {'Stakes':30} | Profit")
    lines.append("─" * 54)

    for br in BANKROLL_TIERS:
        if outcomes:
            res = calc_stakes(outcomes, br)
            stakes_str = " + ".join(f"${s['stake']:.2f}" for s in res["stakes"])
            profit_str = f"+${res['profit']:.2f}"
        else:
            # Fallback for sportsbook arbs that pass raw stake values
            stake_a = arb.get("stake_a", 50) * br / 100
            stake_b = arb.get("stake_b", 50) * br / 100
            profit_val = br * arb["profit_pct"] / 100
            stakes_str = f"${stake_a:.2f} + ${stake_b:.2f}"
            profit_str = f"+${profit_val:.2f}"

        lines.append(f"  ${br:>6} | {stakes_str:<30} | {profit_str}")

    lines.append("")
    lines.append(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)

# ─── DEDUP ────────────────────────────────────────────────────────────────────
alerted_keys: set = set()

def arb_key(arb: dict) -> str:
    return f"{arb['event'][:40]}|{arb['strategy'][:40]}"

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

    # Alert on new ones
    new_arbs = []
    for arb in all_arbs:
        key = arb_key(arb)
        if key not in alerted_keys:
            alerted_keys.add(key)
            new_arbs.append(arb)

    if new_arbs:
        log.info(f"📲 Sending {len(new_arbs)} SMS alert(s)")
        # Send top 3 so you don't get spammed
        for arb in new_arbs[:3]:
            send_sms(format_sms(arb))
        if len(new_arbs) > 3:
            send_sms(f"⚡ +{len(new_arbs)-3} more arbs found this scan. Check dashboard.")
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
