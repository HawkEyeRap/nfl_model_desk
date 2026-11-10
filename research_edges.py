"""
research_edges.py - search for ANY feature set that predicts the market's error (margin - closing line)
out of sample. Walk-forward by season. Reports out-of-sample R^2 / correlation and ATS hit rate.
If nothing beats ~0 reliably, there is no exploitable spread edge in this data.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model import FEATURES

df = pd.read_parquet("data/features.parquet")
p = df[(df.played == 1) & df.spread_line.notna()].copy()
p["resid"] = (p.home_margin - p.spread_line).clip(-30, 30)
p["abs_line"] = p.spread_line.abs()
p["home_dog"] = (p.spread_line < 0).astype(int)
p["big_fav"] = (p.abs_line >= 7).astype(int)
p["total_line_c"] = p.total_line
p["early_season"] = (p.week <= 4).astype(int)
p["post2020"] = (p.season >= 2020).astype(int)
MKT = ["spread_line", "abs_line", "home_dog", "big_fav", "total_line_c", "early_season"]

sets = {
    "market-shape only (line size, home dog, total, early season)": MKT,
    "fundamentals only": FEATURES,
    "fundamentals + market": FEATURES + MKT,
    "injuries + QB only": [c for c in FEATURES if "inj" in c or "qb" in c] + MKT,
}
models = {
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=300.0)),
    "gbm": lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=80, l2_regularization=5.0),
}
print(f"{'feature set':62}{'model':7}{'OOS corr':>9}{'ATS hit (pred>1pt)':>20}{'n':>6}")
for name, cols in sets.items():
    for mname, mk in models.items():
        preds = []
        for S in range(2019, 2027):
            tr, te = p[p.season < S], p[p.season == S]
            m = mk().fit(tr[cols], tr.resid)
            preds.append(pd.Series(m.predict(te[cols]), index=te.index))
        pr = pd.concat(preds)
        d = p.loc[pr.index].assign(pred=pr)
        c = np.corrcoef(d.pred, d.resid)[0, 1]
        s = d[(d.pred.abs() > 1) & (d.resid != 0)]
        hit = ((s.pred > 0) == (s.resid > 0)).mean()
        print(f"{name:62}{mname:7}{c:>+9.3f}{hit:>19.1%}{len(s):>7}")
# simple well-known angles, out of sample 2019-2026
d = p[(p.season >= 2019) & (p.resid != 0)]
print("\nKnown angles 2019-2026 (cover rate of the named side; need 52.4% at -110):")
for nm, mask, side in [("home underdogs", d.spread_line < 0, 1), ("road underdogs", d.spread_line > 0, -1),
                        ("dogs of 7+", d.abs_line >= 7, None), ("divisional dogs", (d.div_game == 1), None),
                        ("team with QB change faded", d.home_qb_change != d.away_qb_change, None)]:
    x = d[mask]
    if side == 1: cov = (x.resid > 0).mean()
    elif side == -1: cov = (x.resid < 0).mean()
    else:
        dog_home = x.spread_line < 0
        cov = np.where(dog_home, x.resid > 0, x.resid < 0).mean() if nm != "team with QB change faded" else \
              np.where(x.home_qb_change == 1, x.resid < 0, x.resid > 0).mean()
    print(f"  {nm:28} {cov:.1%}  (n={len(x)})")
