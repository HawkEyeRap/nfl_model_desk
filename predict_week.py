"""
predict_week.py  -  Step 4: predict the upcoming week, simulate every game, size bets.

Usage:  python predict_week.py [--week 9] [--bankroll 1000]
Output: data/week.json (feeds the dashboard) and picks.csv
"""
import argparse, json, datetime as dt
import numpy as np, pandas as pd
from model import (Fundamental, FEATURES, fit_blend, blended_mean, OutcomePool, simulate, evaluate_bets)
import odds_api

KELLY_FRACTION = 0.25        # quarter-Kelly: standard professional practice (full Kelly is far too volatile)
MAX_BET = 0.02               # never more than 2% of bankroll on one bet
MAX_WEEK = 0.10              # never more than 10% of bankroll in one week
PRICE_EDGE_MIN = 0.02        # line-shopping edge needed vs no-vig consensus

LABELS = {"diff_power": "Recent point-margin rating", "elo_diff": "Elo rating", "diff_qb": "Starting QB quality",
          "diff_qb_change": "QB change", "home_qb_exp": "Home QB experience", "away_qb_exp": "Away QB experience",
          "diff_epa": "EPA per play (opp-adjusted)", "diff_n_epa": "EPA in competitive situations", "diff_sr": "Success rate",
          "diff_n_sr": "Success rate (competitive)", "diff_pass_epa": "Passing efficiency", "diff_rush_epa": "Rushing efficiency",
          "diff_early_epa": "Early-down efficiency", "diff_explosive": "Explosive plays", "diff_sack_rate": "Sack rate",
          "diff_proe": "Pass rate over expected", "diff_td_drive": "TD per drive", "diff_score_drive": "Scores per drive",
          "diff_third_conv": "3rd-down conversion", "diff_rz_td_rate": "Red-zone TD rate", "diff_inj_skill": "Injuries: skill players",
          "diff_inj_ol": "Injuries: offensive line", "diff_inj_dl": "Injuries: defensive line", "diff_inj_lb": "Injuries: linebackers",
          "diff_inj_db": "Injuries: secondary", "diff_inj_starters_out": "Starters out", "home_inj_starters_out": "Home starters out",
          "away_inj_starters_out": "Away starters out", "inj_report_available": "Injury report available", "home_field": "Home field",
          "rest_diff": "Rest advantage", "div_game": "Division game", "thursday": "Thursday game", "primetime": "Prime time",
          "dome": "Dome", "is_post": "Playoffs", "week": "Week of season"}


def drivers(model, row, k=4):
    """Top reasons behind the fundamental model's number (ridge part: coefficient x standardized value)."""
    ridge = model.main[0]
    imp, sc, rg = ridge.named_steps["simpleimputer"], ridge.named_steps["standardscaler"], ridge.named_steps["ridge"]
    x = sc.transform(imp.transform(row[FEATURES].to_frame().T.astype(float)))[0]
    contrib = x * rg.coef_
    contrib[FEATURES.index("inj_report_available")] = 0      # technical flag, not a football reason
    o = np.argsort(-np.abs(contrib))[:k]
    return [{"factor": LABELS.get(FEATURES[i], FEATURES[i]), "pts": round(float(contrib[i]), 1)} for i in o if abs(contrib[i]) >= 0.2]


def injury_list(season, week, team):
    try:
        inj = pd.read_csv(f"raw/inj/inj_{season}.csv", low_memory=False)
    except FileNotFoundError:
        return None, None
    inj["team"] = inj.team.replace({"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"})
    use_week = week if (inj.week == week).any() else int(inj.week.max())
    x = inj[(inj.team == team) & (inj.week == use_week) & inj.report_status.isin(["Out", "Doubtful", "Questionable"])]
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    x = x.assign(o=x.report_status.map(order)).sort_values(["o", "position"])
    return [{"name": r.full_name, "pos": r.position, "status": r.report_status} for r in x.itertuples()], use_week


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int); ap.add_argument("--bankroll", type=float, default=1000)
    a = ap.parse_args()

    df = pd.read_parquet("data/features.parquet")
    summ = json.load(open("data/backtest_summary.json"))
    bt = pd.read_parquet("data/backtest_games.parquet")
    played = df[(df.played == 1) & df.home_margin.notna()]
    season = int(df.season.max())
    frac = df[df.season == season].groupby("week").played.mean()
    week = a.week or int(frac[frac < 0.5].index.min())
    games = df[(df.season == season) & (df.week == week) & (df.played == 0)].copy()
    print(f"Predicting {season} week {week}: {len(games)} games")

    model = Fundamental(n_boot=40).fit(played)
    games["mu_f"], boot = model.predict(games)
    blend = fit_blend(bt)                                  # every out-of-sample season 2020-now
    pool = OutcomePool(played)
    policy = summ["policy"]
    shopping, shop_msg = odds_api.get_shopping()
    print(" ", shop_msg)

    # historical hit rate by confidence bucket (out-of-sample)
    b = bt[bt.home_win != 0.5].assign(pp=lambda d: np.maximum(d.p_home_win, 1 - d.p_home_win),
                                      hit=lambda d: ((d.p_home_win > .5) == (d.home_win == 1)).astype(int))
    buckets = [(0.5, 0.6, "Low"), (0.6, 0.7, "Medium"), (0.7, 0.8, "High"), (0.8, 1.01, "Very high")]
    hist_conf = {lab: {"hit": float(b[(b.pp >= lo) & (b.pp < hi)].hit.mean()), "n": int(((b.pp >= lo) & (b.pp < hi)).sum())}
                 for lo, hi, lab in buckets}

    out, total_stake = [], 0.0
    for i, r in enumerate(games.itertuples()):
        line = r.spread_line
        mu = blended_mean(r.mu_f, line, blend)
        bmu = blended_mean(boot[:, i], line, blend)
        sims = simulate(mu, bmu, blend["beta_se"] * abs(r.mu_f - line), line, r.total_line, pool, n=40000)
        bets = evaluate_bets(r, sims)
        best = max(bets, key=lambda x: x["ev"])
        pick = r.home_team if sims["p_home_win"] >= .5 else r.away_team
        pp = max(sims["p_home_win"], 1 - sims["p_home_win"])
        conf = next(lab for lo, hi, lab in buckets if lo <= pp < hi)
        pol = policy[best["kind"]]
        tier, stake = "PASS", 0.0
        if pol["live"] and best["ev"] >= (pol["ev_min"] or 0):
            tier = "BET"
            stake = a.bankroll * min(KELLY_FRACTION * best["kelly"], MAX_BET)
        elif best["ev"] > 0:
            tier = "WATCH"               # positive on paper, but this bet type never proved itself out of sample
        # line shopping (only if API key configured)
        shop = None
        if shopping and (r.away_team, r.home_team) in shopping:
            s = shopping[(r.away_team, r.home_team)]
            cands = [dict(x, kind="ML") for x in s["ml"][:1]] + [dict(x, kind="Spread") for x in s["spread"][:1]]
            cands = [c for c in cands if c["ev"] >= PRICE_EDGE_MIN]
            shop = {"books": s["books"], "best_ml": s["ml"][:2], "best_spread": s["spread"][:2]}
            if cands and tier != "BET":
                c = max(cands, key=lambda x: x["ev"])
                dec = 1 + c["odds"] / 100 if c["odds"] > 0 else 1 + 100 / abs(c["odds"])
                kel = max(0, (c["fair_p"] * (dec - 1) - (1 - c["fair_p"])) / (dec - 1))
                tier, stake = "PRICE EDGE", a.bankroll * min(KELLY_FRACTION * kel, MAX_BET / 2)
                best = {"kind": c["kind"], "team": c["team"], "odds": c["odds"], "p": c["fair_p"], "ev": c["ev"],
                        "book": c["book"], "point": c.get("point"), "breakeven": 1 / dec, "kelly": kel, "p_push": 0}
        if total_stake + stake > a.bankroll * MAX_WEEK:
            stake = max(0.0, a.bankroll * MAX_WEEK - total_stake)
        total_stake += stake
        injuries = {}
        for t in (r.away_team, r.home_team):
            lst, wk = injury_list(season, week, t)
            injuries[t] = {"players": (lst or [])[:12], "report_week": wk}
        boot_f = np.percentile(boot[:, i], [10, 90])
        out.append({
            "game_id": r.game_id, "kickoff": str(r.kickoff), "day": pd.Timestamp(r.kickoff).strftime("%a %b %d, %I:%M %p ET"),
            "away": r.away_team, "home": r.home_team, "away_qb": r.away_qb_name, "home_qb": r.home_qb_name,
            "spread_line": line, "total_line": r.total_line, "home_ml": r.home_moneyline, "away_ml": r.away_moneyline,
            "home_spread_odds": r.home_spread_odds, "away_spread_odds": r.away_spread_odds,
            "mkt_p_home": r.mkt_p_home,
            "model_margin": round(float(r.mu_f), 1), "model_margin_lo": round(float(boot_f[0]), 1), "model_margin_hi": round(float(boot_f[1]), 1),
            "final_margin": round(float(mu), 1), "pick": pick, "pick_prob": pp, "confidence": conf,
            "p_home_win": sims["p_home_win"], "p_home_win_lo": float(sims["p_home_win_lo"]), "p_home_win_hi": float(sims["p_home_win_hi"]),
            "p_home_cover": sims["p_home_cover"], "p_away_cover": sims["p_away_cover"], "p_push": sims["p_push"],
            "sim_score": f"{r.away_team} {sims['sim_away_pts']:.0f} - {r.home_team} {sims['sim_home_pts']:.0f}",
            "margin_p10": sims["sim_margin_p10"], "margin_p90": sims["sim_margin_p90"], "margin_hist": sims["margin_hist"],
            "bets": bets, "best": best, "tier": tier, "stake": round(stake, 2), "shop": shop,
            "drivers": drivers(model, games.iloc[i]), "injuries": injuries,
            "elo_home": round(float(r.home_elo)), "elo_away": round(float(r.away_elo)),
            "power_home": round(float(r.home_power), 1), "power_away": round(float(r.away_power), 1),
        })

    inj_avail = bool(games.inj_report_available.max() == 1)
    meta = {"season": season, "week": week, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bankroll": a.bankroll, "kelly_fraction": KELLY_FRACTION, "max_bet": MAX_BET, "max_week": MAX_WEEK,
            "injury_report_for_week": inj_avail, "blend": blend, "shopping": shop_msg, "hist_conf": hist_conf,
            "n_train_games": int(len(played)), "total_stake": round(total_stake, 2)}
    json.dump({"meta": meta, "games": out, "backtest": summ}, open("data/week.json", "w"), default=float)
    pd.DataFrame([{"game": f"{g['away']} @ {g['home']}", "day": g["day"], "pick": g["pick"], "win_prob": round(g["pick_prob"], 3),
                   "confidence": g["confidence"], "model_line": g["model_margin"], "vegas_line": g["spread_line"],
                   "tier": g["tier"], "best_bet": f"{g['best']['team']} {g['best']['kind']} ({g['best']['odds']:+.0f})",
                   "ev": round(g["best"]["ev"], 4), "stake": g["stake"]} for g in out]).to_csv("picks.csv", index=False)
    print(pd.read_csv("picks.csv").to_string(index=False))
    print(f"\nTotal stake this week: ${total_stake:.2f} of ${a.bankroll:.0f}.  Injury report for week {week}: "
          f"{'included' if inj_avail else 'NOT yet released - rerun Friday/Saturday'}")


if __name__ == "__main__":
    main()
