# NFL Model Desk

A weekly NFL betting model. Each run builds the data, backtests the system, predicts the upcoming games, sizes the bets, and writes a dashboard.

## Setup (once)
```
pip install pandas numpy scikit-learn pyarrow requests
```
Put every file from this zip in one folder, open a terminal in that folder, and run the commands below. Use `python3` on a Mac.

## Every week
```
python run_all.py                  # full run: about 2 minutes, plus about 5 minutes for the first download
python run_all.py --quick          # skip the backtest (game day)
python run_all.py --bankroll 500   # size bets for your bankroll
python run_all.py --week 10        # a specific week
```
Then open **dashboard.html** in your browser. The same picks are in **picks.csv**.

**When to run it:** Tuesday for an early look, then **Saturday morning** for the version you act on. NFL injury reports come out Wednesday to Friday, and the Saturday run includes them.

## Files
| File | Job |
|---|---|
| `build_data.py` | Downloads about 400k plays from 2016 on, plus injury reports, snap counts, and rosters. Builds opponent-adjusted EPA ratings, QB values, injury burden, and Elo. |
| `model.py` | Football model (ridge regression + gradient boosting, refit 40x on resampled data), market blend, Monte Carlo simulator with key numbers, bet math (EV, Kelly). |
| `backtest.py` | Walk-forward test, 2020–now. Each season is predicted using only earlier seasons. Thresholds are chosen on 2020–22 and checked on 2023–26. |
| `predict_week.py` | Predicts the upcoming week, runs 20,000 sims per game, assigns tiers, and sizes stakes. |
| `odds_api.py` | Optional line shopping across sportsbooks (free key from the-odds-api.com, saved in `odds_api_key.txt`). |
| `make_dashboard.py` + `dashboard_template.html` | Writes `dashboard.html`. |
| `research_edges.py` | Tests whether any feature set or betting angle predicts the market's error. |

## Tiers
- **BET**: the bet type has made money out of sample, and this bet's EV clears the threshold. Stake = quarter-Kelly, capped at 2% of bankroll per bet and 10% per week.
- **WATCH**: positive EV on paper, but this bet type hasn't proven itself. Log it and don't stake money.
- **PRICE EDGE**: one sportsbook's price is off from the sharp no-vig price. Only appears when line shopping is on.
- **PASS**: no edge.

## What the backtest found (Nov 2026)
- Football model alone: 63.8% winners, Brier 0.220. Vegas: 66.6%, Brier 0.211.
- The model's disagreements with the closing line had no predictive value in any season (beta −0.02 to −0.12). So the final numbers match Vegas, and no bet type is live.
- Classic angles (home dogs, big dogs, divisional dogs, fading QB changes) flip between profitable and unprofitable across periods, so they're treated as noise.
- If a future backtest shows a bet type passing both periods, the desk will start recommending it automatically.
