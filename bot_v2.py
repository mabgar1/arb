"""
ArbScanner Bot - Scans Kalshi, Polymarket + sportsbooks for arbitrage.
Sends FREE SMS via Verizon email-to-SMS gateway using Gmail.
"""

import os, time, logging, requests, smtplib, base64, hashlib
from email.mime.text import MIMEText
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
YOUR_PHONE_NUMBER  = os.environ["YOUR_PHONE_NUMBER"]
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY", "")
KALSHI_KEY_ID      = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "")  # full PEM contents
MIN_PROFIT_PCT     = float(os.environ.get("MIN_PROFIT_PCT", "0"))
SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL_SEC", "1800"))

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

# ─── SMS FORMAT ───────────────────────────────────────────────────────────────
def format_sms(arb):
    outcomes = arb["outcomes"]
    pct      = arb["profit_pct"]
    legs     = " | ".join(f"{o['label']}@{o['book']}({o['price']:.2f})" for o in outcomes)
    r100     = calc_stakes(outcomes, 100)
    r500     = calc_stakes(outcomes, 500)
    s100     = "+".join(f"${s['stake']:.0f}" for s in r100["stakes"])
    s500     = "+".join(f"${s['stake']:.0f}" for s in r500["stakes"])
    return (
        f"ARB +{pct:.2f}% | {arb['event'][:38]}\n"
        f"{legs}\n"
        f"$100: {s100} = +${r100['profit']:.2f}\n"
        f"$500: {s500} = +${r500['profit']:.2f}\n"
        f"{datetime.now().strftime('%H:%M')}"
    )[:300]

# ─── KALSHI (RSA-PSS signed requests) ─────────────────────────────────────────
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

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

_kalshi_key = None

def _kalshi_headers(method: str, path: str) -> dict:
    global _kalshi_key
    if _kalshi_key is None:
        _kalshi_key = _load_kalshi_key()
    if _kalshi_key is None:
        return {}
    ts_ms  = str(int(time.time() * 1000))
    # Strip query params from path before signing
    clean_path = path.split("?")[0]
    msg    = (ts_ms + method.upper() + clean_path).encode()
    sig    = _kalshi_key.sign(msg, padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }

def fetch_kalshi():
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY:
        log.warning("No Kalshi credentials — skipping")
        return []
    try:
        path = "/trade-api/v2/markets"
        hdrs = _kalshi_headers("GET", path)
        if not hdrs:
            return []
        resp = requests.get(KALSHI_BASE + "/markets", headers=hdrs, timeout=10,
                            params={"limit":100,"status":"open"})
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        results = []
        for m in markets:
            yes = m.get("yes_ask")
            no  = m.get("no_ask")
            if yes is None or no is None:
                continue
            results.append({
                "source":    "Kalshi",
                "event":     m.get("title", "?"),
                "yes_price": yes / 100,
                "no_price":  no  / 100,
            })
        log.info(f"Kalshi: {len(results)} markets")
        return results
    except Exception as e:
        log.error(f"Kalshi error: {e}")
        return []

# ─── POLYMARKET ───────────────────────────────────────────────────────────────
def fetch_polymarket():
    try:
        resp = requests.get("https://clob.polymarket.com/markets", timeout=10,
                            params={"active":True,"closed":False,"limit":100})
        resp.raise_for_status()
        data    = resp.json()
        markets = data.get("data", data) if isinstance(data, dict) else data
        results = []
        for m in markets:
            tokens  = m.get("tokens", [])
            yes_tok = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
            no_tok  = next((t for t in tokens if t.get("outcome","").upper()=="NO"),  None)
            if not yes_tok or not no_tok:
                continue
            yes_p = float(yes_tok.get("price", 0))
            no_p  = float(no_tok.get("price",  0))
            if yes_p > 0 and no_p > 0:
                results.append({"source":"Polymarket","event":m.get("question","?"),
                                 "yes_price":yes_p,"no_price":no_p})
        log.info(f"Polymarket: {len(results)} markets")
        return results
    except Exception as e:
        log.error(f"Polymarket error: {e}")
        return []

# ─── PREDICTION MARKET CROSS-ARBS ─────────────────────────────────────────────
def find_prediction_arbs(markets_a, markets_b):
    stop = {"the","a","an","of","in","to","vs","and","or","for","will","by","at"}
    arbs = []
    for a in markets_a:
        for b in markets_b:
            wa = set(a["event"].lower().split()) - stop
            wb = set(b["event"].lower().split()) - stop
            if len(wa & wb) < 2:
                continue
            for yes_src, no_src in [(a,b),(b,a)]:
                s = yes_src["yes_price"] + no_src["no_price"]
                if s < 1:
                    profit = (1/s - 1) * 100
                    if 0 < profit <= 15 and profit > MIN_PROFIT_PCT:
                        arbs.append({
                            "event":      a["event"],
                            "profit_pct": profit,
                            "outcomes": [
                                {"label":"YES","book":yes_src["source"],"price":yes_src["yes_price"]},
                                {"label":"NO", "book":no_src["source"], "price":no_src["no_price"]},
                            ]
                        })
    return arbs

# ─── SPORTSBOOKS ──────────────────────────────────────────────────────────────
SPORTS = ["americanfootball_nfl","basketball_nba","baseball_mlb","icehockey_nhl"]
BOOKS  = ["draftkings","fanduel","betmgm","pinnacle","bovada"]

def to_dec(odds):
    return (odds/100)+1 if odds>0 else (100/abs(odds))+1

def fetch_sportsbooks():
    if not ODDS_API_KEY:
        return []
    results = []
    for sport in SPORTS:
        try:
            r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                params={"apiKey":ODDS_API_KEY,"regions":"us","markets":"h2h",
                        "bookmakers":",".join(BOOKS),"oddsFormat":"american"},
                timeout=10)
            r.raise_for_status()
            time.sleep(2)
            for game in r.json():
                bm_data, all_teams = {}, set()
                for bm in game.get("bookmakers",[]):
                    book_odds = {}
                    for mkt in bm.get("markets",[]):
                        if mkt["key"]!="h2h": continue
                        for oc in mkt.get("outcomes",[]):
                            book_odds[oc["name"]] = to_dec(oc["price"])
                            all_teams.add(oc["name"])
                    if book_odds:
                        bm_data[bm["key"]] = book_odds
                all_teams = sorted(all_teams)
                if len(all_teams)>=2 and bm_data:
                    results.append({"event":f"{game.get('away_team')} @ {game.get('home_team')}",
                                    "_teams":all_teams,"_bm":bm_data})
        except Exception as e:
            log.error(f"Odds API ({sport}): {e}")
    log.info(f"Sportsbooks: {len(results)} games")
    return results

def find_sportsbook_arbs(games):
    arbs = []
    for g in games:
        teams = g["_teams"]
        valid = {b:o for b,o in g["_bm"].items() if all(t in o for t in teams)}
        if not valid: continue
        best = {}
        for t in teams:
            bd,bb = max(((o[t],b) for b,o in valid.items()), key=lambda x:x[0])
            best[t] = {"price":bd,"book":bb}
        impl_sum = sum(1/v["price"] for v in best.values())
        if impl_sum >= 1: continue
        profit = (1/impl_sum - 1) * 100
        if profit > 15 or profit <= MIN_PROFIT_PCT: continue
        outcomes = [{"label":t,"book":best[t]["book"],"price":best[t]["price"]} for t in teams]
        arbs.append({"event":g["event"],"profit_pct":profit,"outcomes":outcomes})
    return arbs

# ─── DEDUP ────────────────────────────────────────────────────────────────────
alerted: dict = {}
COOLDOWN = 4 * 3600

def is_new(arb):
    key = arb["event"][:40] + "|" + "|".join(sorted(o["book"] for o in arb["outcomes"]))
    if time.time() - alerted.get(key, 0) > COOLDOWN:
        alerted[key] = time.time()
        return True
    return False

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_scan():
    log.info("═══ Scanning ═══")
    kalshi = fetch_kalshi()
    poly   = fetch_polymarket()
    sb     = fetch_sportsbooks()

    all_arbs = []
    if kalshi and poly:
        pm_arbs = find_prediction_arbs(kalshi, poly)
        log.info(f"Kalshi↔Polymarket: {len(pm_arbs)} arbs")
        all_arbs.extend(pm_arbs)
    if kalshi:
        kk_arbs = find_prediction_arbs(kalshi, kalshi)
        log.info(f"Kalshi internal: {len(kk_arbs)} arbs")
    sb_arbs = find_sportsbook_arbs(sb)
    log.info(f"Sportsbooks: {len(sb_arbs)} arbs")
    all_arbs.extend(sb_arbs)

    all_arbs.sort(key=lambda x: x["profit_pct"], reverse=True)
    new = [a for a in all_arbs if is_new(a)]
    if new:
        log.info(f"📲 Sending {min(3,len(new))} alerts")
        for arb in new[:3]:
            send_sms(format_sms(arb))
            time.sleep(1)
        if len(new) > 3:
            send_sms(f"ARB: +{len(new)-3} more. Best: +{new[0]['profit_pct']:.2f}%")
    else:
        log.info("No new arbs.")
    log.info(f"═══ Done. Next in {SCAN_INTERVAL_SEC}s ═══\n")

def main():
    log.info("ArbScanner starting up...")
    # Test alert so you can confirm texts are working
    test = {
        "event": "TEST - Lakers vs Warriors",
        "profit_pct": 1.87,
        "outcomes": [
            {"label":"Lakers",   "book":"DraftKings", "price":2.10},
            {"label":"Warriors", "book":"Pinnacle",   "price":2.05},
        ]
    }
    send_sms("Bot live! Example alert:\n\n" + format_sms(test))
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
