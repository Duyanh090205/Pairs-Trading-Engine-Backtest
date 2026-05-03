"""
Chunk 5 Validation — Pipeline Assembly & Unit Tests
Run from: week2_signal_engine/
  python validate_chunk5.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import subprocess
import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import compute_half_life, window_from_half_life
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions, count_trades
from src.pipeline.run_week2 import run_pipeline

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")
SIGNALS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "signals")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition

def flag(label, detail=""):
    print(f"  [{WARN}] FLAG: {label}" + (f"  ({detail})" if detail else ""))

all_passed = True

# ── Section 1: Unit tests via pytest ─────────────────────────────────────────
print("\n" + "="*60)
print("CHUNK 5: PIPELINE ASSEMBLY & UNIT TESTS")
print("="*60)

print("\n--- Running pytest (29 unit tests) ---")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd=os.path.dirname(__file__),
)
# Print full pytest output
for line in result.stdout.splitlines():
    print(f"  {line}")
if result.stderr.strip():
    for line in result.stderr.splitlines():
        print(f"  STDERR: {line}")

pytest_passed = result.returncode == 0
all_passed &= check("All unit tests pass", pytest_passed,
                    "see pytest output above" if not pytest_passed else "29/29")

# ── Section 2: run_pipeline() contract checks ─────────────────────────────────
print("\n--- run_pipeline() Contract ---")

cfg      = load_config(CONFIG_PATH)
dates    = cfg["dates"]
ta, tb   = cfg["pairs"]["primary"]

merged, _   = load_pair(DATA_DIR, ta, tb)
formation, trading = split_periods(merged, **dates)

alpha, beta  = estimate_hedge_ratio(formation["close_a"], formation["close_b"])
spread_f     = compute_spread(formation["close_a"], formation["close_b"], alpha, beta)
hl, _        = compute_half_life(spread_f)
window, _    = window_from_half_life(hl)

out = run_pipeline(
    trading["close_a"], trading["close_b"],
    alpha=alpha, beta=beta, window=window,
    entry_z=2.0, exit_z=0.0,
)

all_passed &= check("Output is a DataFrame",
                    isinstance(out, pd.DataFrame))
REQUIRED_COLS = {"spread","rolling_mean","rolling_std","zscore",
                 "position","signal_valid","position_executed"}
all_passed &= check("Output has required columns",
                    REQUIRED_COLS.issubset(out.columns),
                    str(list(out.columns)))
all_passed &= check("Output length matches trading period",
                    len(out) == len(trading),
                    f"{len(out)} rows vs {len(trading)} trading bars")
all_passed &= check("Index matches trading period index",
                    out.index.equals(trading.index))
all_passed &= check("Position column is int8",
                    out["position"].dtype == np.int8,
                    str(out["position"].dtype))
all_passed &= check("Position values are only {-1, 0, +1}",
                    set(out["position"].unique()).issubset({-1, 0, 1}),
                    str(sorted(out["position"].unique())))

n_trades = count_trades(out["position"])
all_passed &= check("Trade count > 0",
                    n_trades > 0, f"{n_trades} trades")

burn_in_nans = int(out["zscore"].isna().sum())
all_passed &= check("Burn-in NaN rows present (rolling window warmup)",
                    burn_in_nans > 0, f"{burn_in_nans} NaN z-score rows at start")

# signal_valid checks
all_passed &= check("signal_valid is bool dtype",
                    out["signal_valid"].dtype == bool,
                    str(out["signal_valid"].dtype))
all_passed &= check("signal_valid is False during entire burn-in",
                    not out["signal_valid"].iloc[:burn_in_nans].any(),
                    f"first {burn_in_nans} rows all False")
all_passed &= check("signal_valid is True after burn-in",
                    out["signal_valid"].iloc[burn_in_nans:].all(),
                    f"all {len(out) - burn_in_nans} post-burn-in rows True")

# position_executed checks
all_passed &= check("position_executed is int8",
                    out["position_executed"].dtype == np.int8,
                    str(out["position_executed"].dtype))
all_passed &= check("position_executed equals position.shift(1) (filled 0)",
                    (out["position_executed"] == out["position"].shift(1).fillna(0).astype("int8")).all(),
                    "exact match on all bars")
all_passed &= check("position_executed[0] is 0 (no prior bar)",
                    int(out["position_executed"].iloc[0]) == 0)

# No-lookahead: formation parameters applied to trading — verify spread is
# computed with the formation alpha/beta, not re-estimated on trading data.
spread_manual = compute_spread(trading["close_a"], trading["close_b"], alpha, beta)
all_passed &= check("Spread matches manual formation-parameter computation",
                    np.allclose(out["spread"].values, spread_manual.values, equal_nan=True))

# ── Section 3: Signal CSV output ──────────────────────────────────────────────
print("\n--- Signal CSV Output ---")

from src.pipeline.run_week2 import main as pipeline_main
pipeline_main()   # runs primary + secondary, saves CSVs

for pair_ta, pair_tb in [cfg["pairs"]["primary"], cfg["pairs"]["secondary"]]:
    csv_path = os.path.join(SIGNALS_DIR, f"signals_{pair_ta}_{pair_tb}.csv")
    exists   = os.path.exists(csv_path)
    all_passed &= check(f"signals_{pair_ta}_{pair_tb}.csv saved", exists, csv_path)

    if exists:
        df_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        all_passed &= check(f"  {pair_ta}/{pair_tb} CSV has all required columns",
                            REQUIRED_COLS.issubset(df_csv.columns),
                            str(sorted(df_csv.columns)))
        all_passed &= check(f"  {pair_ta}/{pair_tb} CSV has > 40000 rows",
                            len(df_csv) > 40_000, f"{len(df_csv):,} rows")

# ── Section 4: No-lookahead invariant end-to-end ─────────────────────────────
print("\n--- No-Lookahead Invariant ---")

# The formation period max timestamp must precede the trading period min timestamp.
form_max = formation.index.max()
trade_min = trading.index.min()
all_passed &= check("Formation ends before trading starts",
                    form_max < trade_min,
                    f"formation_end={form_max}, trading_start={trade_min}")

# run_pipeline must not accept formation prices in this call
# (it receives only trading slice — verify by row count)
all_passed &= check("run_pipeline received only trading-period rows",
                    len(out) == len(trading),
                    f"out={len(out)}, trading={len(trading)}")

# Verify the z-score at bar 0 of trading is NaN (burn-in — no formation leakage)
all_passed &= check("First z-score bar is NaN (no formation leakage into burn-in)",
                    pd.isna(out["zscore"].iloc[0]),
                    f"z[0]={out['zscore'].iloc[0]}")

print("\n" + "="*60)
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Chunk 5 complete. Week 2 engine done.\033[0m")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
