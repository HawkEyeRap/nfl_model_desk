"""
model.py  -  Step 2: the prediction engine.

Three layers, the way professional sports-betting quants structure it:

 1. FUNDAMENTAL MODEL  - predicts the home margin from football data only (no betting lines):
        ensemble of Ridge regression + Gradient-Boosted trees, trained on opponent-adjusted EPA
        ratings, power ratings, Elo, QB value, injuries, rest, travel/situational flags.
        Bootstrapped 25x -> spread of possible answers = parameter uncertainty.

 2. MARKET BLEND       - the betting market is the best single predictor there is, so we don't
        ignore it. We learn (from past seasons only) how much the model's disagreement with the
        line actually predicts results:
              expected margin = Vegas line + beta * (model - Vegas line)
        beta near 0  -> the model adds nothing beyond the line -> no bets.

 3. MONTE CARLO        - simulate each game 20,000 times. Each run draws (a) a plausible true
        expected margin (from the bootstrap spread) and (b) an actual final score from real NFL
        games the market priced the same way, so key numbers (3, 7, 10) and pushes come out right.
        -> P(win), P(cover), P(push), score distribution, and a confidence band on each probability.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RATED = ["epa", "n_epa", "sr", "n_sr", "pass_epa", "rush_epa", "early_epa", "explosive",
         "sack_rate", "proe", "td_drive", "score_drive", "third_conv", "rz_td_rate"]
FEATURES = (["diff_power", "elo_diff", "diff_qb", "diff_qb_change", "home_qb_exp", "away_qb_exp"]
            + [f"diff_{s}" for s in RATED]
            + ["diff_inj_skill", "diff_inj_ol", "diff_inj_dl", "diff_inj_lb", "diff_inj_db",
               "diff_inj_starters_out", "home_inj_starters_out", "away_inj_starters_out", "inj_report_available"]
            + ["home_field", "rest_diff", "div_game", "thursday", "primetime", "dome", "is_post", "week"])
N_SIMS = 20000
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ layer 1: fundamentals
def _ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))


def _gbm(seed=0):
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed)


def _target(df):
    return df.home_margin.clip(-30, 30)       # blowouts beyond 30 carry no extra information


class Fundamental:
    """Ridge + GBM ensemble with bootstrap uncertainty."""

    def __init__(self, n_boot=25):
        self.n_boot = n_boot

    def fit(self, df):
        X, y = df[FEATURES], _target(df)
        self.main = [_ridge().fit(X, y), _gbm().fit(X, y)]
        self.boot = []
        for b in range(self.n_boot):
            idx = RNG.integers(0, len(df), len(df))
            Xb, yb = X.iloc[idx], y.iloc[idx]
            self.boot.append([_ridge().fit(Xb, yb), _gbm(b + 1).fit(Xb, yb)])
        return self

    def predict(self, df):
        X = df[FEATURES]
        mu = np.mean([m.predict(X) for m in self.main], axis=0)
        boot = np.array([np.mean([m.predict(X) for m in ms], axis=0) for ms in self.boot])  # (B, n)
        return mu, boot


# ------------------------------------------------------------------ layer 2: market blend
def fit_blend(oos):
    """
    oos: past games with out-of-sample fundamental predictions 'mu_f'.
    Regress (actual margin - line) on (model - line). Returns beta, standard error, intercept.
    """
    d = oos.dropna(subset=["mu_f", "spread_line", "home_margin"])
    x = (d.mu_f - d.spread_line).values
    y = (d.home_margin - d.spread_line).values
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid.var(ddof=2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    beta = float(np.clip(coef[1], 0, 1))     # can't trust the model MORE than 100% or negatively
    return {"beta": beta, "beta_raw": float(coef[1]), "beta_se": float(se[1]), "n": len(d),
            "intercept": float(coef[0])}


def blended_mean(mu_f, line, blend):
    return line + blend["beta"] * (mu_f - line)


# ------------------------------------------------------------------ layer 3: Monte Carlo
class OutcomePool:
    """
    Final-margin distribution for a game, built the way sports-betting quants do it:
      * base shape: Normal(center, sigma)  (Stern 1991; sigma ~13.5 fitted on past games)
      * key-number weights: each whole-number margin is multiplied by how much more (or less)
        often it really happens than a smooth curve predicts (3 and 7 are ~2-2.5x, 0 is ~0.1x)
      * efficient-market anchor: for a line L we find the center c*(L) at which both sides cover
        equally often. That's the market's view. The model's disagreement shifts the center from c*.
    """
    K = np.arange(-80, 81)

    def __init__(self, hist):
        h = hist.dropna(subset=["spread_line", "home_margin", "total_line", "game_total"])
        resid = (h.home_margin - h.spread_line).values
        self.sigma = float(np.clip(resid.std(), 12, 15))
        absm = np.abs(h.home_margin.values).astype(int)
        emp = np.bincount(np.clip(absm, 0, 80), minlength=81).astype(float)
        emp = emp / emp.sum()
        # smooth reference: average of neighbours excluding the point itself
        pad = np.pad(emp, 3, mode="edge")
        smooth = np.array([(pad[i:i + 7].sum() - pad[i + 3]) / 6 for i in range(81)])
        w = np.where(smooth > 0, emp / np.maximum(smooth, 1e-9), 1.0)
        w[20:] = np.clip(w[20:], 0.5, 2.0)          # shrink noisy weights in the tail
        w[40:] = 1.0
        self.w = np.clip(w, 0.05, 4.0)
        self.wk = self.w[np.abs(self.K)]
        self.tres = (h.game_total - h.total_line).values
        self._anchor = {}

    def pmf(self, center):
        p = np.exp(-0.5 * ((self.K - center) / self.sigma) ** 2) * self.wk
        return p / p.sum()

    def anchor(self, line):
        """Center at which home and away cover equally (the market's fair center)."""
        key = round(float(line), 1)
        if key in self._anchor:
            return self._anchor[key]
        lo, hi = line - 6, line + 6
        for _ in range(40):
            mid = (lo + hi) / 2
            p = self.pmf(mid)
            if p[self.K > line].sum() > p[self.K < line].sum(): hi = mid
            else: lo = mid
        self._anchor[key] = (lo + hi) / 2
        return self._anchor[key]

    def sample(self, centers, line):
        """centers: simulated expected home margins (line + model edge); returns integer margins."""
        c0 = self.anchor(line)
        eff = c0 + (centers - line)                         # edge applied around the fair center
        grid = np.round(eff * 4) / 4                        # 0.25-pt grid for speed
        out = np.empty(len(centers))
        for g in np.unique(grid):
            idx = np.where(grid == g)[0]
            cdf = np.cumsum(self.pmf(g))
            out[idx] = self.K[np.searchsorted(cdf, RNG.random(len(idx)) * cdf[-1])]
        return out, RNG.choice(self.tres, len(centers))


def simulate(mu, boot, sigma_extra, line, total_line, pool, n=N_SIMS):
    """
    mu: blended expected margin; boot: bootstrap expected margins (already blended);
    returns dict of probabilities + intervals + score summary for one game.
    """
    # parameter uncertainty: draw from the bootstrap distribution (+ small extra spread for model error)
    centers = RNG.choice(boot, n) + RNG.normal(0, sigma_extra, n)
    margins, tres = pool.sample(centers, line if np.isfinite(line) else 0.0)
    total = (total_line if np.isfinite(total_line) else 44.0) + tres
    home_pts = np.clip(np.round((total + margins) / 2), 0, None)
    away_pts = np.clip(home_pts - margins, 0, None)
    out = {
        "p_home_win": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
        "p_tie": float(np.mean(margins == 0)),
        "p_home_cover": float(np.mean(margins > line)) if np.isfinite(line) else np.nan,
        "p_push": float(np.mean(margins == line)) if np.isfinite(line) else np.nan,
        "p_away_cover": float(np.mean(margins < line)) if np.isfinite(line) else np.nan,
        "sim_margin_p10": float(np.percentile(margins, 10)), "sim_margin_p50": float(np.median(margins)),
        "sim_margin_p90": float(np.percentile(margins, 90)),
        "sim_home_pts": float(np.median(home_pts)), "sim_away_pts": float(np.median(away_pts)),
    }
    # confidence band on the probabilities: rerun a smaller sim per bootstrap center
    pw, pc = [], []
    for b in boot:
        m, _ = pool.sample(np.full(1500, b), line if np.isfinite(line) else 0.0)
        pw.append(np.mean(m > 0) + 0.5 * np.mean(m == 0))
        if np.isfinite(line):
            pc.append(np.mean(m > line) / max(1e-9, 1 - np.mean(m == line)))
    out["p_home_win_lo"], out["p_home_win_hi"] = np.percentile(pw, [10, 90])
    if pc:
        out["p_home_cover_ex_push_lo"], out["p_home_cover_ex_push_hi"] = np.percentile(pc, [10, 90])
    # histogram of margins for the dashboard (-35..35)
    hist, _ = np.histogram(margins, bins=np.arange(-35.5, 36.5, 1))   # beyond +/-35 left off the chart
    out["margin_hist"] = (hist / n).round(4).tolist()
    return out


# ------------------------------------------------------------------ betting math
def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / np.abs(odds))


def bet_ev(p_win, p_push, odds):
    """Expected profit per $1 staked."""
    dec = american_to_decimal(odds)
    p_loss = 1 - p_win - p_push
    return p_win * (dec - 1) - p_loss


def kelly_fraction(p_win, p_push, odds):
    b = american_to_decimal(odds) - 1
    p_loss = 1 - p_win - p_push
    return np.maximum(0, (b * p_win - p_loss) / b)


def evaluate_bets(row, sims):
    """All four bets for one game -> list of dicts with EV and Kelly."""
    bets = []
    tie = sims["p_tie"]
    if np.isfinite(row.home_moneyline) and np.isfinite(row.away_moneyline):
        ph = sims["p_home_win"] - 0.5 * tie            # ML ties = push
        pa = 1 - sims["p_home_win"] - 0.5 * tie
        bets.append(("ML", row.home_team, row.home_moneyline, ph, tie))
        bets.append(("ML", row.away_team, row.away_moneyline, pa, tie))
    if np.isfinite(row.spread_line):
        ho = row.home_spread_odds if np.isfinite(row.home_spread_odds) else -110
        ao = row.away_spread_odds if np.isfinite(row.away_spread_odds) else -110
        bets.append(("Spread", row.home_team, ho, sims["p_home_cover"], sims["p_push"]))
        bets.append(("Spread", row.away_team, ao, sims["p_away_cover"], sims["p_push"]))
    out = []
    for kind, team, odds, p, push in bets:
        out.append({"kind": kind, "team": team, "odds": float(odds), "p": float(p), "p_push": float(push),
                    "breakeven": float(1 / american_to_decimal(odds)),
                    "ev": float(bet_ev(p, push, odds)), "kelly": float(kelly_fraction(p, push, odds))})
    return out
