"""
backtest.py  -  Step 3: honest walk-forward test of the whole system, including betting.

For each season S:
   - train the fundamental model ONLY on seasons before S
   - learn the market-blend weight (beta) ONLY from out-of-sample predictions on seasons before S
   - simulate every game of S, compute EV for every moneyline/spread bet at the real posted odds
   - record what would have happened
Then betting thresholds are CHOSEN on 2020-2022 and VERIFIED on 2023-2026 (never seen while choosing).
"""
import json, time
import numpy as np, pandas as pd
from model import Fundamental, fit_blend, blended_mean, OutcomePool, simulate, evaluate_bets

FIRST_OOS, FIRST_BET = 2018, 2020
SELECT = range(2020, 2023)         # seasons used to pick thresholds
HOLDOUT = range(2023, 2027)        # seasons used to verify them
EV_GRID = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]


def settle(bet, r):
    """Profit per $1 for a settled bet."""
    from model import american_to_decimal
    dec = float(american_to_decimal(bet["odds"]))
    m = r.home_margin
    home = bet["team"] == r.home_team
    if bet["kind"] == "ML":
        if m == 0: return 0.0
        won = (m > 0) == home
    else:
        if m == r.spread_line: return 0.0
        won = (m > r.spread_line) == home
    return dec - 1 if won else -1.0


def main(n_sims=4000):
    t0 = time.time()
    df = pd.read_parquet("data/features.parquet")
    played = df[(df.played == 1) & df.home_margin.notna()].copy()

    # 1) out-of-sample fundamental predictions, season by season
    oos = []
    models = {}
    for S in range(FIRST_OOS, int(played.season.max()) + 1):
        tr = played[played.season < S]
        m = Fundamental(n_boot=20).fit(tr)
        te = played[played.season == S].copy()
        te["mu_f"], boot = m.predict(te)
        te["boot_idx"] = range(len(te))
        models[S] = (te, boot)
        oos.append(te[["game_id", "season", "mu_f", "spread_line", "home_margin"]])
        print(f"  fundamental OOS {S}: trained on {len(tr)} games  ({time.time()-t0:.0f}s)")
    oos = pd.concat(oos)

    # 2) simulate + bet every game from FIRST_BET on
    rows, betlog, blends = [], [], {}
    for S in range(FIRST_BET, int(played.season.max()) + 1):
        blend = fit_blend(oos[oos.season < S]); blends[S] = blend
        pool = OutcomePool(played[played.season < S])
        te, boot = models[S]
        for r in te.itertuples():
            line = r.spread_line
            mu = blended_mean(r.mu_f, line, blend)
            bmu = blended_mean(boot[:, r.boot_idx], line, blend)
            extra = blend["beta_se"] * abs(r.mu_f - line)
            sims = simulate(mu, bmu, extra, line, r.total_line, pool, n=n_sims)
            rows.append({"game_id": r.game_id, "season": S, "week": r.week, "home_team": r.home_team, "away_team": r.away_team,
                         "home_margin": r.home_margin, "home_win": r.home_win, "spread_line": line, "mkt_p_home": r.mkt_p_home,
                         "mu_f": r.mu_f, "mu": mu, **{k: v for k, v in sims.items() if k != "margin_hist"}})
            for b in evaluate_bets(r, sims):
                b.update(game_id=r.game_id, season=S, week=r.week, profit=settle(b, r))
                betlog.append(b)
        print(f"  simulated {S}: beta={blend['beta']:.3f} (raw {blend['beta_raw']:+.3f} ± {blend['beta_se']:.3f})")
    games = pd.DataFrame(rows)
    bets = pd.DataFrame(betlog)
    games.to_parquet("data/backtest_games.parquet"); bets.to_parquet("data/backtest_bets.parquet")

    # 3) accuracy vs the market
    g = games[games.home_win != 0.5]
    y = g.home_win.values

    def brier(p): return float(np.mean((p - y) ** 2))
    def logloss(p): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = {
        "n_games": int(len(g)),
        "model_only_brier": brier(1 / (1 + np.exp(-g.mu_f / 6.5))),   # fundamentals alone, logistic on margin
        "blended_brier": brier(g.p_home_win.values), "market_brier": brier(g.mkt_p_home.values),
        "blended_logloss": logloss(g.p_home_win.values), "market_logloss": logloss(g.mkt_p_home.values),
        "blended_acc": float(np.mean((g.p_home_win > .5) == y)), "market_acc": float(np.mean((g.mkt_p_home > .5) == y)),
        "model_only_acc": float(np.mean((g.mu_f > 0) == y)),
        "model_only_mae": float(np.mean(np.abs(games.mu_f - games.home_margin))),
        "market_mae": float(np.mean(np.abs(games.spread_line - games.home_margin))),
        "blended_mae": float(np.mean(np.abs(games.mu - games.home_margin))),
    }
    # calibration table for the win probabilities
    bins = np.linspace(0, 1, 11)
    g2 = g.assign(bin=pd.cut(g.p_home_win, bins))
    calib = g2.groupby("bin", observed=True).agg(pred=("p_home_win", "mean"), actual=("home_win", "mean"), n=("home_win", "size")).reset_index(drop=True)

    # 4) betting: best-EV bet per game, by threshold, by bet type
    best = bets.sort_values("ev", ascending=False).drop_duplicates(["game_id", "kind"])

    def perf(sub):
        n = len(sub)
        if n == 0: return {"n": 0, "roi": None, "win_rate": None, "units": 0.0}
        settled = sub[sub.profit != 0]
        return {"n": int(n), "roi": float(sub.profit.mean()), "units": float(sub.profit.sum()),
                "win_rate": float((settled.profit > 0).mean()) if len(settled) else None,
                "se": float(sub.profit.std(ddof=1) / np.sqrt(n)) if n > 1 else None}

    grid = []
    for kind in ["ML", "Spread"]:
        for T in EV_GRID:
            sel = best[(best.kind == kind) & (best.ev > T)]
            grid.append({"kind": kind, "ev_min": T, "select": perf(sel[sel.season.isin(SELECT)]),
                         "holdout": perf(sel[sel.season.isin(HOLDOUT)]), "all": perf(sel)})
    policy = {}
    for kind in ["ML", "Spread"]:
        cands = [x for x in grid if x["kind"] == kind and x["select"]["n"] >= 40]
        pick = max(cands, key=lambda x: x["select"]["roi"]) if cands else None
        live = bool(pick and pick["select"]["roi"] > 0 and pick["holdout"]["n"] >= 30 and pick["holdout"]["roi"] > 0)
        policy[kind] = {"ev_min": pick["ev_min"] if pick else None, "live": live,
                        "select": pick["select"] if pick else None, "holdout": pick["holdout"] if pick else None}
    by_season = {}
    for kind in ["ML", "Spread"]:
        T = policy[kind]["ev_min"] or 0
        s = best[(best.kind == kind) & (best.ev > T)]
        by_season[kind] = {int(k): perf(v) for k, v in s.groupby("season")}

    # cumulative units for the chart: chosen-threshold bets, flat 1 unit, on one shared week-by-week axis
    weeks = games[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    curves = {"label": [f"{a}-W{b}" for a, b in zip(weeks.season, weeks.week)]}
    for kind in ["ML", "Spread"]:
        T = policy[kind]["ev_min"] or 0
        s = best[(best.kind == kind) & (best.ev > T)].groupby(["season", "week"]).profit.sum()
        curves[kind] = weeks.set_index(["season", "week"]).join(s).profit.fillna(0).cumsum().round(2).tolist()
    summary = {"accuracy": acc, "calibration": calib.round(3).to_dict("records"), "grid": grid, "policy": policy,
               "by_season": by_season, "curves": curves,
               "blends": {int(k): v for k, v in blends.items()}, "select_seasons": list(SELECT), "holdout_seasons": list(HOLDOUT)}
    json.dump(summary, open("data/backtest_summary.json", "w"), indent=1, default=float)

    print(f"\nACCURACY ({acc['n_games']} games, {FIRST_BET}-{int(played.season.max())}, all out-of-sample)")
    print(f"  Brier      fundamentals {acc['model_only_brier']:.4f} | blended {acc['blended_brier']:.4f} | market {acc['market_brier']:.4f}")
    print(f"  Log loss   blended {acc['blended_logloss']:.4f} | market {acc['market_logloss']:.4f}")
    print(f"  Accuracy   fundamentals {acc['model_only_acc']:.1%} | blended {acc['blended_acc']:.1%} | market {acc['market_acc']:.1%}")
    print(f"  Margin MAE fundamentals {acc['model_only_mae']:.2f} | blended {acc['blended_mae']:.2f} | market {acc['market_mae']:.2f}")
    print("\nBETTING (best-EV side per game, flat 1 unit)")
    for x in grid:
        s, h = x["select"], x["holdout"]
        f = lambda d: f"{d['n']:4d} bets ROI {d['roi']:+6.1%}" if d["n"] else "   0 bets"
        print(f"  {x['kind']:6} EV>{x['ev_min']:.0%}:  2020-22 {f(s)}   |   2023-26 {f(h)}")
    print("\nPOLICY:", json.dumps({k: {"ev_min": v["ev_min"], "live": v["live"]} for k, v in policy.items()}))
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
