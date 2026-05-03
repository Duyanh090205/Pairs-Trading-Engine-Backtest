"""
Chunk 4 Validation — Diagnostics, Sensitivity, Visuals
Run from: week2_signal_engine/
  python validate_chunk4.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import (
    compute_half_life, hurst_exponent, window_from_half_life,
)
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions
from src.analytics.diagnostics import empirical_coverage, tail_behavior, quantile_crossval
from src.analytics.sensitivity import run_sensitivity_analysis
from src.visuals.plots import plot_signal_validation, plot_rolling_hurst, plot_qq_normal

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "figures")

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

# ── Build primary pair pipeline ───────────────────────────────────────────────
cfg   = load_config(CONFIG_PATH)
dates = cfg["dates"]
ta, tb = cfg["pairs"]["primary"]

merged, _ = load_pair(DATA_DIR, ta, tb)
formation, trading = split_periods(merged, **dates)

fa, fb = formation["close_a"], formation["close_b"]
alpha, beta  = estimate_hedge_ratio(fa, fb)
spread_f     = compute_spread(fa, fb, alpha, beta)
hl, _        = compute_half_life(spread_f)
H_form       = hurst_exponent(spread_f)
window, _    = window_from_half_life(hl)

# Formation Z-scores (for adaptive threshold + cross-val)
zdf_f    = compute_zscore(spread_f, window=window)
z_f      = zdf_f["zscore"].dropna()
adaptive = round(float(z_f.abs().quantile(0.95)), 4)

# Trading period
spread_t = compute_spread(trading["close_a"], trading["close_b"], alpha, beta)
zdf_t    = compute_zscore(spread_t, window=window)
z_t      = zdf_t["zscore"]

pos_fixed = generate_positions(z_t, entry_z=2.0,    exit_z=0.0)
pos_adapt = generate_positions(z_t, entry_z=adaptive, exit_z=0.0)

# Full pipeline DataFrame for charting
df_pipeline = trading[["close_a", "close_b"]].copy()
df_pipeline["spread"]       = spread_t
df_pipeline["rolling_mean"] = zdf_t["rolling_mean"]
df_pipeline["rolling_std"]  = zdf_t["rolling_std"]
df_pipeline["zscore"]       = z_t
df_pipeline["position"]     = pos_fixed

print("\n" + "="*60)
print(f"CHUNK 4: DIAGNOSTICS — PRIMARY PAIR {ta}/{tb}")
print("="*60)

# ── Section 1: Empirical coverage ─────────────────────────────────────────────
print("\n--- Empirical Coverage Table ---")
cov_df = empirical_coverage(z_t)
print(f"\n  {'Threshold':>10}  {'Normal %':>9}  {'Empirical %':>12}  {'Gap':>7}")
print(f"  {'-'*45}")
for _, row in cov_df.iterrows():
    gap_str = f"{row['gap_pct']:+.2f}%"
    print(f"  {row['threshold']:>10}  {row['normal_pct']:>8.2f}%  "
          f"{row['empirical_pct']:>11.2f}%  {gap_str:>7}")

# The 2-sigma coverage check — primary diagnostic
cov_2s = cov_df.loc[cov_df["threshold"] == "+-2.0", "empirical_pct"].values[0]
gap_2s = cov_df.loc[cov_df["threshold"] == "+-2.0", "gap_pct"].values[0]
all_passed &= check("Coverage table has 5 rows", len(cov_df) == 5)
all_passed &= check("+-2.0 empirical coverage is a valid percentage",
                    0 < cov_2s < 100, f"{cov_2s:.2f}%")
if gap_2s > 0:
    flag(f"Fat tails confirmed at +-2.0: normal={cov_df.iloc[2]['normal_pct']:.1f}%, "
         f"empirical={cov_2s:.1f}%, gap={gap_2s:+.2f}%",
         "More extreme Z-scores than normal theory predicts (expected on 1-min data)")
else:
    flag(f"Thinner-than-normal tails at +-2.0: gap={gap_2s:+.2f}%",
         "Spread may be over-dispersed or window too short")

# ── Section 2: Tail behaviour ─────────────────────────────────────────────────
print("\n--- Tail Behaviour ---")
tb_stats = tail_behavior(z_t)
for k, v in tb_stats.items():
    print(f"  {k:20s}: {v}")

all_passed &= check("n_obs > 40000 (sufficient trading bars)",
                    tb_stats["n_obs"] > 40_000, f"n={tb_stats['n_obs']:,}")
all_passed &= check("std in plausible range (0.5, 2.0)",
                    0.5 < tb_stats["std"] < 2.0,
                    f"std={tb_stats['std']}")
if tb_stats["std"] > 1.1:
    flag(f"Z-score std = {tb_stats['std']:.4f} (> 1.0)",
         f"Window ({window} bars) is only {window/hl*100:.0f}% of half-life ({hl:.0f} bars). "
         f"rolling_std underestimates true long-run spread variance, inflating Z-scores. "
         f"This is the mechanical cause of the fat-tail coverage result above. "
         f"Excess kurtosis={tb_stats['excess_kurtosis']:.3f} (near-zero) confirms no genuine fat tails.")

if abs(tb_stats["skewness"]) > 0.5:
    flag(f"Skewness = {tb_stats['skewness']:.3f}",
         "Asymmetric Z-score distribution — spread may drift directionally")
if tb_stats["excess_kurtosis"] > 3:
    flag(f"Excess kurtosis = {tb_stats['excess_kurtosis']:.2f}",
         "Fat tails confirmed — connect to LTCM reading in Signal Logic Document")

# ── Section 3: Quantile cross-validation ──────────────────────────────────────
print("\n--- Quantile Cross-Validation (formation vs trading) ---")
xval = quantile_crossval(z_f, z_t)
for k, v in xval.items():
    print(f"  {k:20s}: {v}")

all_passed &= check("Cross-val returns expected keys",
                    all(k in xval for k in ["formation_q95","trading_q95","abs_drift","regime_shift"]))
if xval["regime_shift"]:
    flag(f"Regime shift detected: formation_q95={xval['formation_q95']}, "
         f"trading_q95={xval['trading_q95']}, drift={xval['abs_drift']:.4f}",
         "Adaptive threshold calibrated on a different volatility regime")
else:
    print(f"  [  OK  ] No significant regime shift (drift={xval['abs_drift']:.4f} < 0.5)")

# ── Section 4: Sensitivity analysis ──────────────────────────────────────────
print("\n--- Threshold Sensitivity ---")
thresholds = sorted([1.5, 2.0, adaptive, 2.5, 3.0])
sens_df = run_sensitivity_analysis(spread_t, window=window, thresholds=thresholds)
print(f"\n  {'Threshold':>10}  {'Trades':>8}  {'Trades/Day':>11}")
print(f"  {'-'*33}")
for _, row in sens_df.iterrows():
    marker = " <-- adaptive" if abs(row["threshold"] - adaptive) < 0.001 else ""
    print(f"  {row['threshold']:>10.4f}  {row['n_trades']:>8}  "
          f"{row['trades_per_day']:>10.3f}{marker}")

all_passed &= check("Sensitivity table has correct number of rows",
                    len(sens_df) == len(thresholds),
                    f"{len(sens_df)} rows")
all_passed &= check("Trade count decreases as threshold increases",
                    sens_df["n_trades"].is_monotonic_decreasing,
                    f"counts: {list(sens_df['n_trades'])}")

# ── Section 5: Rolling Hurst ──────────────────────────────────────────────────
print("\n--- Rolling Hurst ---")

def rolling_hurst_series(spread, rolling_window=5000, step=500, max_lag=100):
    from src.analytics.characterize import hurst_exponent
    vals = spread.dropna()
    rows = []
    for start in range(0, len(vals) - rolling_window, step):
        end = start + rolling_window
        H   = hurst_exponent(vals.iloc[start:end], max_lag=max_lag)
        rows.append({"hurst": H})
    df_h = pd.DataFrame(rows)
    # Assign timestamps at each window end
    timestamps = [vals.index[min(start + rolling_window - 1, len(vals)-1)]
                  for start in range(0, len(vals) - rolling_window, step)]
    df_h.index = pd.DatetimeIndex(timestamps[:len(df_h)])
    return df_h

hurst_df = rolling_hurst_series(spread_t)
pct_mr   = (hurst_df["hurst"] < 0.5).mean() * 100
print(f"  Rolling windows computed : {len(hurst_df)}")
print(f"  % windows with H < 0.5  : {pct_mr:.1f}%")
all_passed &= check("Rolling Hurst produced at least 5 windows",
                    len(hurst_df) >= 5, f"{len(hurst_df)} windows")
if pct_mr < 40:
    flag(f"Only {pct_mr:.1f}% of rolling windows show H < 0.5",
         "Mean-reversion assumption is weak in the trading period")

# ── Section 6: Plots ──────────────────────────────────────────────────────────
print("\n--- Generating Plots ---")

try:
    fig1 = plot_signal_validation(
        df_pipeline, ta, tb, beta=beta, window=window, entry_z=2.0,
        save_path=os.path.join(FIG_DIR, "signal_validation.png"),
    )
    all_passed &= check("signal_validation.png saved",
                        os.path.exists(os.path.join(FIG_DIR, "signal_validation.png")))
except Exception as e:
    print(f"  [{FAIL}] signal_validation plot failed: {e}")
    all_passed = False

try:
    fig2 = plot_rolling_hurst(
        hurst_df, ta, tb,
        save_path=os.path.join(FIG_DIR, "rolling_hurst.png"),
    )
    all_passed &= check("rolling_hurst.png saved",
                        os.path.exists(os.path.join(FIG_DIR, "rolling_hurst.png")))
except Exception as e:
    print(f"  [{FAIL}] rolling_hurst plot failed: {e}")
    all_passed = False

try:
    fig3 = plot_qq_normal(
        z_t, ta, tb,
        save_path=os.path.join(FIG_DIR, "qq_plot.png"),
    )
    all_passed &= check("qq_plot.png saved",
                        os.path.exists(os.path.join(FIG_DIR, "qq_plot.png")))
except Exception as e:
    print(f"  [{FAIL}] QQ plot failed: {e}")
    all_passed = False

print("\n" + "="*60)
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Chunk 4 complete.\033[0m")
    print(f"  Figures saved to: {FIG_DIR}")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
