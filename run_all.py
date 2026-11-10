"""
run_all.py - one command for the whole weekly workflow.

    python run_all.py                    # full run (~2 min): download -> features -> backtest -> predict -> dashboard
    python run_all.py --quick            # skip the backtest (reuses the last one) - use on game day
    python run_all.py --bankroll 500     # size bets for a $500 bankroll
    python run_all.py --week 10          # predict a specific week
"""
import subprocess, sys, time

args = sys.argv[1:]
quick = "--quick" in args
passthru = [a for a in args if a != "--quick"]


def step(name, cmd):
    t = time.time()
    print(f"\n=== {name} ===")
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"Step failed: {name}")
    print(f"({time.time()-t:.0f}s)")


step("1/4 Download + build features", ["build_data.py"])
if not quick:
    step("2/4 Walk-forward backtest", ["backtest.py"])
step("3/4 Predict + simulate + size bets", ["predict_week.py"] + passthru)
step("4/4 Build dashboard", ["make_dashboard.py"])
print("\nDone. Open dashboard.html in your browser. Picks are also in picks.csv.")
