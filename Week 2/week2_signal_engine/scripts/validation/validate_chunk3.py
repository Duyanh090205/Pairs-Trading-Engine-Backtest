"""
Chunk 3 Validation Script — Z-Score Engine + State Machine
Run from: week2_signal_engine/
  python validate_chunk3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import compute_half_life, hurst_exponent, window_from_half_life
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_signals, generate_positions, count_trades

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")

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

# ── Load config and build pair list ───────────────────────────────────────────
cfg   = load_config(CONFIG_PATH)
dates = cfg["dates"]
eng   = cfg["engine"]

all_pairs = [("primary", tuple(cfg["pairs"]["primary"])),
             ("secondary", tuple(cfg["pairs"]["secondary"]))]
for i, p in enumerate(cfg["pairs"]["alternatives"]):
    all_pairs.append((f"alt_{i+1}", tuple(p)))
for i, p in enumerate(cfg["pairs"]["benchmarks"]):
    all_pairs.append((f"benchmark_{i+1}", tuple(p)))
for i, p in enumerate(cfg["pairs"]["negative_controls"]):
    all_pairs.append((f"neg_ctrl_{i+1}", tuple(p)))

# ── Section 1: Unit tests on zscore and state machine ─────────────────────────
print("\n" + "="*60)
print("SECTION 1: UNIT TESTS")
print("="*60)

print("\n--- Z-Score ---")
import pandas as pd
rng   = np.random.default_rng(42)
dummy = pd.Series(rng.normal(0, 1, 1000), name="spread")

zdf   = compute_zscore(dummy, window=60)
all_passed &= check("Returns DataFrame with 3 columns",
                    list(zdf.columns) == ["rolling_mean", "rolling_std", "zscore"])
all_passed &= check("First 29 rows are NaN (min_periods = window//2 = 30)",
                    zdf["zscore"].iloc[:29].isna().all(),
                    f"first non-NaN at index {zdf['zscore'].first_valid_index()}")
all_passed &= check("After warm-up, zscore values are finite",
                    zdf["zscore"].dropna().apply(np.isfinite).all())
all_passed &= check("Rolling std uses ddof=1 (std > 0 on non-constant series)",
                    (zdf["rolling_std"].dropna() > 0).all())

# Constant series -> std == 0 -> eps kicks in, zscore should be 0 (not inf)
const = pd.Series(np.ones(200), name="spread")
zdf_c = compute_zscore(const, window=60)
all_passed &= check("eps guard: constant spread gives zscore=0, not Inf",
                    zdf_c["zscore"].dropna().abs().max() < 1e-5,
                    f"max |z| = {zdf_c['zscore'].dropna().abs().max():.2e}")

print("\n--- State Machine ---")
# Test 1: single excursion crosses entry then exits at zero
z_test1 = np.array([0.0, 0.5, 1.0, 2.5, 2.0, 1.0, 0.5, -0.1, -0.5], dtype=np.float64)
pos1 = generate_positions(pd.Series(z_test1), entry_z=2.0, exit_z=0.0)
all_passed &= check("Entry: flat until z > 2.0",
                    pos1.iloc[2] == 0 and pos1.iloc[3] == -1,
                    f"pos@z=1.0={pos1.iloc[2]}, pos@z=2.5={pos1.iloc[3]}")
all_passed &= check("Exit: short cleared when z crosses 0 (z=-0.1)",
                    pos1.iloc[7] == 0,
                    f"pos@z=-0.1={pos1.iloc[7]}")
all_passed &= check("No re-entry while still above entry_z boundary",
                    (pos1.iloc[3:7] == -1).all(),
                    f"positions during excursion: {list(pos1.iloc[3:7])}")

# Test 2: long entry, exit
z_test2 = np.array([0.0, -2.5, -1.5, -0.5, 0.1, 2.5, 1.0, 0.0, -0.1], dtype=np.float64)
pos2 = generate_positions(pd.Series(z_test2), entry_z=2.0, exit_z=0.0)
all_passed &= check("Long entry fires when z < -2.0",
                    pos2.iloc[1] == 1, f"pos@z=-2.5={pos2.iloc[1]}")
all_passed &= check("Long exit when z >= 0",
                    pos2.iloc[4] == 0, f"pos@z=0.1={pos2.iloc[4]}")
all_passed &= check("Re-entry into short after exiting long",
                    pos2.iloc[5] == -1, f"pos@z=2.5={pos2.iloc[5]}")

# Test 3: NaN handling — state must be held through NaN bars
z_test3 = np.array([np.nan, np.nan, 2.5, np.nan, np.nan, 0.0, -0.1], dtype=np.float64)
pos3 = generate_positions(pd.Series(z_test3), entry_z=2.0, exit_z=0.0)
all_passed &= check("NaN bars before entry -> position stays 0",
                    pos3.iloc[0] == 0 and pos3.iloc[1] == 0)
all_passed &= check("NaN bars after entry -> position holds -1",
                    pos3.iloc[3] == -1 and pos3.iloc[4] == -1,
                    f"positions: {list(pos3)}")
all_passed &= check("Exit after NaN stretch works correctly",
                    pos3.iloc[6] == 0, f"pos@z=-0.1={pos3.iloc[6]}")

# Test 4: count_trades
z_test4 = pd.Series([0.0, 2.5, 1.0, 0.0, -0.1, -2.5, -1.0, 0.1], dtype=np.float64)
pos4 = generate_positions(z_test4, entry_z=2.0, exit_z=0.0)
n4   = count_trades(pos4)
all_passed &= check("count_trades: 2 excursions = 2 trades", n4 == 2, f"counted {n4}")

# Test 5: positions dtype
all_passed &= check("Position dtype is int8",
                    pos1.dtype == np.int8, f"dtype={pos1.dtype}")
all_passed &= check("Position values only in {-1, 0, 1}",
                    set(pos1.unique()).issubset({-1, 0, 1}))

# ── Section 2: Full pipeline across all 11 pairs ──────────────────────────────
print("\n" + "="*60)
print("SECTION 2: FULL PIPELINE — ALL 11 PAIRS")
print("="*60)

results = []
errors  = []

for label, (ta, tb) in all_pairs:
    try:
        merged, _ = load_pair(DATA_DIR, ta, tb)
        formation, trading = split_periods(merged, **dates)

        fa, fb = formation["close_a"], formation["close_b"]
        alpha, beta    = estimate_hedge_ratio(fa, fb)
        spread_f       = compute_spread(fa, fb, alpha, beta)
        hl, _          = compute_half_life(spread_f)
        window, _      = window_from_half_life(hl)

        # Adaptive threshold: abs(Z) 95th percentile on formation
        zdf_f     = compute_zscore(spread_f, window=window)
        z_f_clean = zdf_f["zscore"].dropna()
        adaptive  = round(float(z_f_clean.abs().quantile(0.95)), 4)

        # Run pipeline on trading period (hedge ratio from formation — no lookahead)
        spread_t  = compute_spread(trading["close_a"], trading["close_b"], alpha, beta)
        zdf_t     = compute_zscore(spread_t, window=window)
        z_t       = zdf_t["zscore"]

        raw_sig   = generate_signals(z_t, entry_z=2.0)
        pos_fixed = generate_positions(z_t, entry_z=2.0,    exit_z=0.0)
        pos_adapt = generate_positions(z_t, entry_z=adaptive, exit_z=0.0)

        n_fixed   = count_trades(pos_fixed)
        n_adapt   = count_trades(pos_adapt)
        n_days    = trading.index.normalize().nunique()

        # Spot checks
        ok_dtype   = pos_fixed.dtype == np.int8
        ok_values  = set(pos_fixed.unique()).issubset({-1, 0, 1})
        ok_nan_start = z_t.iloc[:window // 2 - 1].isna().all()
        pct_in_pos = (pos_fixed != 0).mean() * 100

        results.append({
            "label":      label,
            "pair":       f"{ta}/{tb}",
            "window":     window,
            "adaptive":   adaptive,
            "n_fixed":    n_fixed,
            "n_adapt":    n_adapt,
            "n_days":     n_days,
            "tpd_fixed":  round(n_fixed / n_days, 3),
            "pct_in_pos": round(pct_in_pos, 1),
            "ok_dtype":   ok_dtype,
            "ok_values":  ok_values,
            "ok_nan":     ok_nan_start,
        })
    except Exception as e:
        errors.append((label, f"{ta}/{tb}", str(e)))

# Print results and run checks
print(f"\n{'Label':12s} {'Pair':12s} {'Win':>4} {'Adapt':>6} "
      f"{'Fixed±2':>8} {'Adapt':>7} {'TPD-fix':>8} {'%InPos':>7} {'OK?':>5}")
print("-" * 75)

for r in results:
    ok_all = r["ok_dtype"] and r["ok_values"] and r["ok_nan"]
    flag_str = "OK" if ok_all else "FAIL"
    all_passed &= ok_all
    print(f"  {r['label']:12s} {r['pair']:12s} {r['window']:>4} {r['adaptive']:>6.3f} "
          f"{r['n_fixed']:>8} {r['n_adapt']:>7} {r['tpd_fixed']:>8.3f} "
          f"{r['pct_in_pos']:>6.1f}% {flag_str:>5}")

# Specific pair flags
print("\n--- Flags ---")
for r in results:
    ta, tb = r["pair"].split("/")
    if r["n_fixed"] == 0:
        flag(f"{r['pair']}: ZERO trades with fixed ±2.0 threshold",
             "Z-score may never reach ±2 — check spread volatility")
    if r["pct_in_pos"] > 50:
        flag(f"{r['pair']}: {r['pct_in_pos']:.1f}% of trading bars in position",
             "State machine may be stuck in a position — check slow-exit conditions")
    if r["tpd_fixed"] > 5:
        flag(f"{r['pair']}: {r['tpd_fixed']:.1f} trades/day with fixed threshold",
             "Very high frequency — check if window is too short relative to half-life")

# Sanity: benchmark should have far more trades than candidates
bench = next((r for r in results if r["label"] == "benchmark_1"), None)
primary = next((r for r in results if r["label"] == "primary"), None)
if bench and primary:
    all_passed &= check(
        "GOOG/GOOGL (benchmark) has more trades than CMS/DUK (primary)",
        bench["n_fixed"] > primary["n_fixed"],
        f"GOOG/GOOGL={bench['n_fixed']}, CMS/DUK={primary['n_fixed']}"
    )

# Negative controls: should have trades (engine fires) but drift should be noted
neg1 = next((r for r in results if r["label"] == "neg_ctrl_1"), None)
if neg1:
    print(f"\n  Note: neg_ctrl_1 CVNA/ISRG fires {neg1['n_fixed']} trades on fixed ±2.0 --")
    print( "        expected (engine runs mechanically), but spread drift is large.")
    print( "        These are NOT valid signals -- useful as a cautionary document example.")

if errors:
    print(f"\n--- ERRORS ({len(errors)}) ---")
    for label, pair, err in errors:
        print(f"  {label:15s} {pair:12s}  ERROR: {err}")
    all_passed = False

print("\n" + "="*60)
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Chunk 3 complete.\033[0m")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
