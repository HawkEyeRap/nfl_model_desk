"""
build_data.py  -  Step 1: download + engineer every feature.

Sources (all free, from nflverse):
  games.csv            schedule, scores, Vegas spread / moneyline / total + the juice, weather, starting QBs
  play_by_play (pbp)   every snap since 2016 (~500k plays): EPA, success, pass-rate-over-expected, ...
  injuries             official weekly injury reports (Out / Doubtful / Questionable)
  snap_counts          share of snaps each player plays  -> how important an injured player is
  weekly_rosters       ID map between the injury file and the snap file

Output: data/features.parquet  (one row per game, every feature uses ONLY information
        available before that game's week kicked off)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FIRST, LAST = 2016, 2026
REL = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
RAW, OUT = "raw", "data"
OFFLINE = "--offline" in sys.argv
CURRENT = LAST                     # always re-download the current season (new games, new injuries)


def fetch(url, path, always=False):
    if os.path.exists(path) and (OFFLINE or not always):
        return path
    import requests
    for i in range(3):
        try:
            r = requests.get(url, timeout=300); r.raise_for_status()
            open(path, "wb").write(r.content); return path
        except Exception as e:
            print("  retry", url, e); time.sleep(3)
    if os.path.exists(path):
        return path
    raise RuntimeError(f"could not download {url}")


def download():
    for d in ["pbp", "inj", "snap", "ros"]:
        os.makedirs(f"{RAW}/{d}", exist_ok=True)
    fetch(GAMES_URL, f"{RAW}/games.csv", always=True)
    for y in range(FIRST, LAST + 1):
        cur = y == CURRENT
        fetch(f"{REL}/pbp/play_by_play_{y}.parquet", f"{RAW}/pbp/pbp_{y}.parquet", cur)
        fetch(f"{REL}/injuries/injuries_{y}.csv", f"{RAW}/inj/inj_{y}.csv", cur)
        fetch(f"{REL}/snap_counts/snap_counts_{y}.csv", f"{RAW}/snap/snap_{y}.csv", cur)
        fetch(f"{REL}/weekly_rosters/roster_weekly_{y}.csv", f"{RAW}/ros/ros_{y}.csv", cur)


# ----------------------------------------------------------------------------- games
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}


def ml_to_prob(ml):
    return np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))


def load_games():
    g = pd.read_csv(f"{RAW}/games.csv")
    g = g[g.season.between(FIRST, LAST)].copy()
    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(TEAM_FIX)                      # one code per franchise across seasons
    g["played"] = g.home_score.notna().astype(int)
    g["home_margin"] = g.home_score - g.away_score
    g["game_total"] = g.home_score + g.away_score
    g["home_win"] = np.where(g.played == 0, np.nan, np.where(g.home_margin > 0, 1, np.where(g.home_margin < 0, 0, 0.5)))
    g["kickoff"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"))
    ph, pa = ml_to_prob(g.home_moneyline), ml_to_prob(g.away_moneyline)
    g["mkt_p_home"] = ph / (ph + pa)                       # no-vig market win probability
    g["is_post"] = (g.game_type != "REG").astype(int)
    g["primetime"] = (pd.to_datetime(g.gametime, format="%H:%M", errors="coerce").dt.hour >= 19).astype(int)
    g["thursday"] = (g.weekday == "Thursday").astype(int)
    g["dome"] = g.roof.isin(["dome", "closed"]).astype(int)
    g["rest_diff"] = g.home_rest - g.away_rest
    g["neutral"] = (g.location == "Neutral").astype(int)
    # a global week index so ages can be measured across seasons (offseason = ~30 idle weeks)
    g["wk_idx"] = (g.season - FIRST) * 52 + g.week
    return g.sort_values("kickoff").reset_index(drop=True)


# ----------------------------------------------------------------------------- play-by-play
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "pass", "rush",
            "qb_dropback", "qb_epa", "epa", "success", "wp", "down", "yards_gained", "interception",
            "fumble_lost", "sack", "pass_oe", "cpoe", "third_down_converted", "third_down_failed",
            "fixed_drive", "fixed_drive_result", "drive_inside20", "touchdown", "penalty_yards",
            "penalty_team", "passer_player_id", "special_teams_play", "qb_kneel", "qb_spike"]


def team_game_metrics():
    """One row per (game, offense): efficiency stats. Defense = the opponent's offense row."""
    rows, qb_rows = [], []
    for y in range(FIRST, LAST + 1):
        p = pd.read_parquet(f"{RAW}/pbp/pbp_{y}.parquet", columns=PBP_COLS)
        p = p[p.posteam.notna()]
        p["posteam"] = p.posteam.replace(TEAM_FIX); p["defteam"] = p.defteam.replace(TEAM_FIX)
        p["penalty_team"] = p.penalty_team.replace(TEAM_FIX)
        plays = p[(p["pass"] == 1) | (p["rush"] == 1)]
        plays = plays[(plays.qb_kneel != 1) & (plays.qb_spike != 1) & plays.epa.notna()]
        neutral = plays[plays.wp.between(0.1, 0.9)]          # drop garbage time (standard analytics practice)
        k = ["game_id", "posteam"]
        a = plays.groupby(k).agg(
            plays=("epa", "size"), epa=("epa", "mean"), sr=("success", "mean"),
            explosive=("yards_gained", lambda s: (s >= 20).mean()),
            int_rate=("interception", "mean"), fum_rate=("fumble_lost", "mean"))
        n = neutral.groupby(k).agg(n_epa=("epa", "mean"), n_sr=("success", "mean"), proe=("pass_oe", "mean"))
        ps = neutral[neutral["pass"] == 1].groupby(k).agg(pass_epa=("epa", "mean"), sack_rate=("sack", "mean"), cpoe=("cpoe", "mean"))
        ru = neutral[neutral["rush"] == 1].groupby(k).agg(rush_epa=("epa", "mean"), rush_sr=("success", "mean"))
        ed = neutral[neutral.down <= 2].groupby(k).agg(early_epa=("epa", "mean"))
        td3 = p[p.down == 3].groupby(k).agg(c=("third_down_converted", "sum"), f=("third_down_failed", "sum"))
        td3 = (td3.c / (td3.c + td3.f)).rename("third_conv").to_frame()
        dr = p.drop_duplicates(["game_id", "posteam", "fixed_drive"])
        dr = dr.assign(td=dr.fixed_drive_result.eq("Touchdown"),
                       score=dr.fixed_drive_result.isin(["Touchdown", "Field goal"]),
                       rz=dr.drive_inside20.eq(1))
        d = dr.groupby(k).agg(drives=("td", "size"), td_drive=("td", "mean"), score_drive=("score", "mean"),
                              rz_trips=("rz", "sum"), rz_td=("td", lambda s: s[dr.loc[s.index, "rz"]].sum()))
        d["rz_td_rate"] = d.rz_td / d.rz_trips.replace(0, np.nan)
        st = p[p.special_teams_play == 1].groupby(k).agg(st_epa=("epa", "sum"))
        pen = p[p.penalty_team.notna()].groupby(["game_id", "penalty_team"]).penalty_yards.sum().rename("pen_yds")
        pen.index.names = k
        t = a.join([n, ps, ru, ed, td3, d[["drives", "td_drive", "score_drive", "rz_td_rate"]], st, pen]).reset_index()
        t = t.rename(columns={"posteam": "team"})
        t["season"] = y
        rows.append(t)
        # QB efficiency by game (for starter ratings)
        q = p[(p.qb_dropback == 1) & p.qb_epa.notna() & p.passer_player_id.notna()]
        qb_rows.append(q.groupby(["game_id", "passer_player_id"]).agg(qb_epa=("qb_epa", "sum"), dropbacks=("qb_epa", "size")).reset_index())
        print(f"  pbp {y}: {len(plays):,} plays")
    tg = pd.concat(rows, ignore_index=True)
    qb = pd.concat(qb_rows, ignore_index=True)
    return tg, qb


# ----------------------------------------------------------------------------- opponent-adjusted ratings
RATED = ["epa", "n_epa", "sr", "n_sr", "pass_epa", "rush_epa", "early_epa", "explosive",
         "sack_rate", "proe", "td_drive", "score_drive", "third_conv", "rz_td_rate"]


def opponent_adjusted_ratings(games, tg, halflife=10.0, alpha=4.0):
    """
    For every week, fit a weighted ridge regression on all earlier games:
        stat(offense, game) = offense_strength[team] + defense_strength[opponent] + home edge
    Weights decay with age (half-life in weeks; the offseason counts as idle weeks, so old seasons fade).
    This is the 'adjusted EPA' approach used by sharp NFL modelers: a team that played strong
    defenses isn't penalized, a team that feasted on weak ones isn't over-rated.
    Also fits a points-margin power rating (like SRS with recency decay).
    """
    from sklearn.linear_model import Ridge
    played = games[games.played == 1][["game_id", "home_team", "away_team", "home_margin", "wk_idx", "neutral"]]
    long = tg.merge(games[["game_id", "home_team", "away_team", "wk_idx", "neutral"]], on="game_id")
    long["opp"] = np.where(long.team == long.home_team, long.away_team, long.home_team)
    long["is_home"] = np.where(long.neutral == 1, 0, (long.team == long.home_team).astype(int) * 2 - 1)
    teams = sorted(set(games.home_team) | set(games.away_team))
    ti = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    out = []
    for (season, week), wk in games.groupby(["season", "week"]):
        cut = wk.wk_idx.min()
        hist = long[long.wk_idx < cut]
        if len(hist) < 200:
            continue
        w = 0.5 ** ((cut - hist.wk_idx) / halflife)
        X = np.zeros((len(hist), 2 * nT + 1))
        X[np.arange(len(hist)), hist.team.map(ti)] = 1
        X[np.arange(len(hist)), nT + hist.opp.map(ti)] = 1
        X[:, -1] = hist.is_home
        rec = {"season": season, "week": week}
        res = {t: dict(rec, team=t) for t in teams}
        for s in RATED:
            y = hist[s].values
            ok = ~np.isnan(y)
            mu = np.average(y[ok], weights=w[ok])
            m = Ridge(alpha=alpha, fit_intercept=False).fit(X[ok], y[ok] - mu, sample_weight=w[ok])
            for t in teams:
                res[t][f"o_{s}"] = m.coef_[ti[t]]          # offense: higher = better (except sack_rate)
                res[t][f"d_{s}"] = m.coef_[nT + ti[t]]     # defense: what opponents do vs this team (lower = better)
        # margin power rating
        ph = played[played.wk_idx < cut]
        wm = 0.5 ** ((cut - ph.wk_idx) / halflife)
        Xm = np.zeros((len(ph), nT + 1))
        Xm[np.arange(len(ph)), ph.home_team.map(ti)] = 1
        Xm[np.arange(len(ph)), ph.away_team.map(ti)] = -1
        Xm[:, -1] = 1 - ph.neutral
        mm = Ridge(alpha=2.0, fit_intercept=False).fit(Xm, ph.home_margin.clip(-28, 28), sample_weight=wm)
        for t in teams:
            res[t]["power"] = mm.coef_[ti[t]]
            res[t]["hfa_est"] = mm.coef_[-1]
        out += list(res.values())
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Elo
def elo_ratings(games, k=20, hfa=48, revert=1 / 3):
    elo, season, pre = {}, None, []
    for r in games.itertuples():
        if r.season != season:
            elo = {t: 1500 + (e - 1500) * (1 - revert) for t, e in elo.items()}; season = r.season
        h, a = elo.get(r.home_team, 1500), elo.get(r.away_team, 1500)
        pre.append((h, a))
        if r.played == 1:
            d0 = h + (0 if r.neutral else hfa) - a
            exp = 1 / (1 + 10 ** (-d0 / 400))
            res = r.home_win
            mult = np.log(abs(r.home_margin) + 1) * 2.2 / ((d0 if res == 1 else -d0) * .001 + 2.2)
            d = k * mult * (res - exp)
            elo[r.home_team], elo[r.away_team] = h + d, a - d
    games["home_elo"], games["away_elo"] = zip(*pre)
    games["elo_diff"] = games.home_elo - games.away_elo
    return games


# ----------------------------------------------------------------------------- QBs
def qb_ratings(games, qb, k_shrink=250, prior=-0.05):
    """Each starter's EPA per dropback before this game, shrunk toward a backup-level prior."""
    qb = qb.merge(games[["game_id", "kickoff"]], on="game_id").sort_values("kickoff")
    g = qb.groupby("passer_player_id")
    # recency-weighted running totals (each past game decays 2% per game played)
    decay = 0.98
    vals = {}
    out = []
    for r in qb.itertuples():
        e, n = vals.get(r.passer_player_id, (0.0, 0.0))
        vals[r.passer_player_id] = (e * decay + r.qb_epa, n * decay + r.dropbacks)
        out.append((r.passer_player_id, r.kickoff, *vals[r.passer_player_id]))
    hist = pd.DataFrame(out, columns=["qb_id", "kickoff", "cum_epa", "cum_db"]).sort_values("kickoff")
    res = []
    for side in ["home", "away"]:
        x = games[["game_id", "kickoff", f"{side}_qb_id"]].rename(columns={f"{side}_qb_id": "qb_id"}).dropna().sort_values("kickoff")
        x = pd.merge_asof(x, hist, on="kickoff", by="qb_id", allow_exact_matches=False)
        x[f"{side}_qb_val"] = (x.cum_epa.fillna(0) + prior * k_shrink) / (x.cum_db.fillna(0) + k_shrink)
        x[f"{side}_qb_exp"] = x.cum_db.fillna(0)
        res.append(x[["game_id", f"{side}_qb_val", f"{side}_qb_exp"]])
    for r in res:
        games = games.merge(r, on="game_id", how="left")
    # QB change flag: is this starter different from the team's most common starter in its last 4 games?
    long = pd.concat([games[["game_id", "kickoff", "home_team", "home_qb_id"]].set_axis(["game_id", "kickoff", "team", "qb"], axis=1),
                      games[["game_id", "kickoff", "away_team", "away_qb_id"]].set_axis(["game_id", "kickoff", "team", "qb"], axis=1)])
    long = long.sort_values("kickoff")
    long["prev_qb"] = long.groupby("team").qb.shift(1)
    long["qb_change"] = ((long.qb != long.prev_qb) & long.prev_qb.notna() & long.qb.notna()).astype(int)
    for side in ["home", "away"]:
        games = games.merge(long[["game_id", "team", "qb_change"]].rename(columns={"team": f"{side}_team", "qb_change": f"{side}_qb_change"}),
                            on=["game_id", f"{side}_team"], how="left")
    return games


# ----------------------------------------------------------------------------- injuries
POS_UNIT = {"QB": "qb", "RB": "skill", "FB": "skill", "WR": "skill", "TE": "skill",
            "T": "ol", "G": "ol", "C": "ol", "OT": "ol", "OG": "ol", "OL": "ol",
            "DE": "dl", "DT": "dl", "NT": "dl", "DL": "dl", "LB": "lb", "ILB": "lb", "OLB": "lb", "MLB": "lb",
            "CB": "db", "S": "db", "SS": "db", "FS": "db", "DB": "db", "K": "st", "P": "st", "LS": "st"}


def injury_features():
    """
    For each team-week: total 'snap share' of players listed Out/Doubtful (and 25% weight for Questionable),
    split by unit. A player's importance = his average snap share in that season's earlier weeks
    (so a starting LT out matters ~10x more than a backup TE).
    """
    out = []
    for y in range(FIRST, LAST + 1):
        inj = pd.read_csv(f"{RAW}/inj/inj_{y}.csv", low_memory=False)
        inj["team"] = inj.team.replace(TEAM_FIX)
        snap = pd.read_csv(f"{RAW}/snap/snap_{y}.csv", low_memory=False)
        ros = pd.read_csv(f"{RAW}/ros/ros_{y}.csv", low_memory=False, usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("gsis_id")
        snap["share"] = snap[["offense_pct", "defense_pct"]].max(axis=1)
        snap = snap.sort_values("week")
        # importance entering week w = mean share over weeks < w (expanding, shifted)
        snap["imp"] = snap.groupby("pfr_player_id").share.transform(lambda s: s.shift(1).expanding().mean())
        # players with no earlier games this season: use their first-game share as a weak proxy (0.5x)
        snap["imp"] = snap.imp.fillna(snap.share * 0.5)
        imp = snap[["pfr_player_id", "week", "imp"]]
        inj = inj.merge(ros, on="gsis_id", how="left").rename(columns={"pfr_id": "pfr_player_id"})
        # use the player's importance from the latest week <= report week
        inj = inj.sort_values("week")
        imp = imp.sort_values("week")
        inj = pd.merge_asof(inj, imp, on="week", by="pfr_player_id", direction="backward")
        inj["imp"] = inj.imp.fillna(0.15)          # unknown player: assume small role
        wgt = inj.report_status.map({"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.25}).fillna(0)
        inj["lost"] = inj.imp * wgt
        inj["unit"] = inj.position.map(POS_UNIT).fillna("other")
        inj["starter_out"] = ((inj.imp >= 0.6) & (wgt >= 0.85)).astype(int)
        agg = inj.pivot_table(index=["season", "week", "team"], columns="unit", values="lost", aggfunc="sum", fill_value=0)
        agg.columns = [f"inj_{c}" for c in agg.columns]
        agg["inj_starters_out"] = inj.groupby(["season", "week", "team"]).starter_out.sum()
        agg["inj_total"] = agg[[c for c in agg.columns if c.startswith("inj_") and c not in ("inj_starters_out", "inj_qb")]].sum(axis=1)
        out.append(agg.reset_index())
    df = pd.concat(out, ignore_index=True).fillna(0)
    df["inj_report_available"] = 1
    return df


# ----------------------------------------------------------------------------- assemble
def main():
    t0 = time.time()
    if not OFFLINE:
        print("Downloading (only the current season is re-fetched each run)...")
    download()
    games = load_games()
    print(f"games: {len(games):,} ({games.played.sum():,} played)")
    tg, qb = team_game_metrics()
    tg.to_parquet(f"{OUT}/team_game_metrics.parquet")
    print("opponent-adjusted ratings (one ridge fit per week)...")
    rat = opponent_adjusted_ratings(games, tg)
    games = elo_ratings(games)
    games = qb_ratings(games, qb)
    inj = injury_features()

    for side in ["home", "away"]:
        r = rat.rename(columns={c: f"{side}_{c}" for c in rat.columns if c not in ("season", "week", "team")})
        games = games.merge(r.rename(columns={"team": f"{side}_team"}), on=["season", "week", f"{side}_team"], how="left")
        i = inj.rename(columns={c: f"{side}_{c}" for c in inj.columns if c not in ("season", "week", "team")})
        games = games.merge(i.rename(columns={"team": f"{side}_team"}), on=["season", "week", f"{side}_team"], how="left")
    # weeks where the injury file exists but a team had nobody listed -> 0 injuries
    have = inj.groupby(["season", "week"]).size().rename("wk_has_inj").reset_index()
    games = games.merge(have, on=["season", "week"], how="left")
    icols = [c for c in games.columns if "_inj_" in c]
    games.loc[games.wk_has_inj.notna(), icols] = games.loc[games.wk_has_inj.notna(), icols].fillna(0)
    games["inj_report_available"] = games.wk_has_inj.notna().astype(int)

    # home-minus-away differences (offense vs the other team's defense matchups)
    for s in RATED:
        games[f"x_{s}_home_off_vs_away_def"] = games[f"home_o_{s}"] + games[f"away_d_{s}"]
        games[f"x_{s}_away_off_vs_home_def"] = games[f"away_o_{s}"] + games[f"home_d_{s}"]
        games[f"diff_{s}"] = games[f"x_{s}_home_off_vs_away_def"] - games[f"x_{s}_away_off_vs_home_def"]
    games["diff_power"] = games.home_power - games.away_power
    games["diff_qb"] = games.home_qb_val - games.away_qb_val
    games["diff_qb_change"] = games.home_qb_change - games.away_qb_change
    for u in ["skill", "ol", "dl", "lb", "db", "total", "starters_out"]:
        a, b = f"home_inj_{u}", f"away_inj_{u}"
        if a in games and b in games:
            games[f"diff_inj_{u}"] = games[a] - games[b]
    games["home_field"] = 1 - games.neutral
    games.to_parquet(f"{OUT}/features.parquet")
    print(f"features: {games.shape[0]:,} games x {games.shape[1]} columns  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
