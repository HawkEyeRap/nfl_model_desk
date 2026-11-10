"""
odds_api.py  -  OPTIONAL line shopping across sportsbooks (The Odds API, free tier = 500 requests/month).

Why: the model can't beat the closing line, but different sportsbooks post different prices for
the same game. Professional '+EV' bettors compare each book's price to the no-vig CONSENSUS of
all books (or to the sharpest book, Pinnacle) and bet only when one book is clearly off-market.
That edge comes from arithmetic, not from predicting football.

Setup:  get a free key at https://the-odds-api.com  then either
        export ODDS_API_KEY=yourkey      (Mac/Linux)     or     set ODDS_API_KEY=yourkey   (Windows)
        or put the key in a file called odds_api_key.txt next to this script.
"""
import os, json
import numpy as np

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
SHARP = ["pinnacle", "lowvig", "betonlineag"]          # books whose prices are closest to 'true' odds


def api_key():
    k = os.environ.get("ODDS_API_KEY")
    if not k and os.path.exists("odds_api_key.txt"):
        k = open("odds_api_key.txt").read().strip()
    return k or None


def fetch(key):
    import requests
    r = requests.get(URL, params={"apiKey": key, "regions": "us,us2,eu", "markets": "h2h,spreads",
                                  "oddsFormat": "american"}, timeout=30)
    r.raise_for_status()
    json.dump(r.json(), open("data/odds_api_latest.json", "w"))
    return r.json(), r.headers.get("x-requests-remaining")


def to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def shop(events):
    """
    Returns {(away, home): {"ml": [...], "spread": [...]}} where each entry is a book price with
    its edge versus the no-vig fair probability (sharp books if available, else median consensus).
    """
    out = {}
    for ev in events:
        home, away = NAME_TO_ABBR.get(ev.get("home_team")), NAME_TO_ABBR.get(ev.get("away_team"))
        if not home or not away:
            continue
        ml, sp = [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                oc = {NAME_TO_ABBR.get(o["name"]): o for o in mk.get("outcomes", [])}
                if home not in oc or away not in oc:
                    continue
                if mk["key"] == "h2h":
                    ml.append((bk["key"], bk["title"], oc[home]["price"], oc[away]["price"]))
                elif mk["key"] == "spreads":
                    sp.append((bk["key"], bk["title"], oc[home].get("point"), oc[home]["price"], oc[away].get("point"), oc[away]["price"]))
        res = {"ml": [], "spread": [], "books": len({b[0] for b in ml + sp})}
        if ml:
            fair = _fair([(h, a) for k, _, h, a in ml if k in SHARP] or [(h, a) for _, _, h, a in ml])
            for k, title, h, a in ml:
                res["ml"].append({"book": title, "team": home, "odds": h, "fair_p": fair[0], "ev": fair[0] * (_dec(h) - 1) - (1 - fair[0])})
                res["ml"].append({"book": title, "team": away, "odds": a, "fair_p": fair[1], "ev": fair[1] * (_dec(a) - 1) - (1 - fair[1])})
        if sp:
            # compare only books posting the consensus (most common) home line
            pts = [x[2] for x in sp if x[2] is not None]
            if pts:
                main = max(set(pts), key=pts.count)
                same = [x for x in sp if x[2] == main]
                fair = _fair([(x[3], x[5]) for x in same if x[0] in SHARP] or [(x[3], x[5]) for x in same])
                for k, title, hp, ho, ap, ao in same:
                    res["spread"].append({"book": title, "team": home, "point": hp, "odds": ho, "fair_p": fair[0], "ev": fair[0] * (_dec(ho) - 1) - (1 - fair[0])})
                    res["spread"].append({"book": title, "team": away, "point": ap, "odds": ao, "fair_p": fair[1], "ev": fair[1] * (_dec(ao) - 1) - (1 - fair[1])})
        for m in ("ml", "spread"):
            res[m].sort(key=lambda x: -x["ev"])
        out[(away, home)] = res
    return out


def _dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def _fair(pairs):
    """No-vig probability: remove each book's margin, then take the median across books."""
    ph = [to_prob(h) / (to_prob(h) + to_prob(a)) for h, a in pairs]
    p = float(np.median(ph))
    return p, 1 - p


def get_shopping():
    key = api_key()
    if not key:
        return None, "No Odds API key set: line shopping is off (see odds_api.py to turn it on)."
    try:
        ev, remaining = fetch(key)
        return shop(ev), f"Line shopping on: {len(ev)} games, requests left this month: {remaining}"
    except Exception as e:
        return None, f"Odds API error: {e}"
