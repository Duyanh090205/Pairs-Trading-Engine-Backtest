"""
Kalman Filter Validation & Static vs Kalman Comparison
Run from: week2_signal_engine/
  python validate_kalman.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.signals.kalman import kalman_hedge_ratio
from src.analytics.characterize import compute_half_life, hurst_exponent, window_from_half_life
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions, count_trades
from src.analytics.diagnostics import empirical_coverage, tail_behavior, quantile_crossval
from src.visuals.plots import plot_kalman_beta

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "figures")
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

# ── Load data ─────────────────────────────────────────────────────────────────
cfg    = load_config(CONFIG_PATH)
dates  = cfg["dates"]
delta  = float(cfg.get("engine", {}).get("kalman_delta", 1e-5))
ta, tb = cfg["pairs"]["primary"]

merged, _   = load_pair(DATA_DIR, ta, tb)
formation, trading = split_periods(merged, **dates)

print("\n" + "="*60)
print(f"KALMAN FILTER VALIDATION  --  {ta}/{tb}")
print(f"delta = {delta}  ({len(merged):,} total bars)")
print("="*60)

# ── Section 1: Kalman estimation on full year ─────────────────────────────────
print("\n--- Section 1: Kalman Estimation ---")

alpha_full, beta_full, kf_diag = kalman_hedge_ratio(
    merged["close_a"], merged["close_b"], delta=delta
)

all_passed &= check("alpha_series length matches merged", len(alpha_full) == len(merged))
all_passed &= check("beta_series length matches merged",  len(beta_full)  == len(merged))
all_passed &= check("No NaN in alpha_series", not alpha_full.isna().any())
all_passed &= check("No NaN in beta_series",  not beta_full.isna().any())
all_passed &= check("Final beta is positive",  kf_diag["final_beta"] > 0,
                    f"final_beta={kf_diag['final_beta']:.6f}")
all_passed &= check("Final beta in plausible range (0.3 to 5.0)",
                    0.3 < kf_diag["final_beta"] < 5.0,
                    f"{kf_diag['final_beta']:.4f}")

for k, v in kf_diag.items():
    print(f"    {k:20s}: {v}")

# ── Section 2: Static OLS baseline ───────────────────────────────────────────
print("\n--- Section 2: Static OLS Baseline ---")

alpha_ols, beta_ols = estimate_hedge_ratio(formation["close_a"], formation["close_b"])
print(f"    alpha_OLS = {alpha_ols:.6f}")
print(f"    beta_OLS  = {beta_ols:.6f}")

beta_drift_form = float(beta_full.loc[formation.index].iloc[-1] - beta_ols)
beta_drift_full = float(beta_full.iloc[-1] - beta_ols)
print(f"    Kalman beta at formation_end : {beta_full.loc[formation.index].iloc[-1]:.6f}  "
      f"(drift vs OLS: {beta_drift_form:+.6f})")
print(f"    Kalman beta at year_end      : {beta_full.iloc[-1]:.6f}  "
      f"(drift vs OLS: {beta_drift_full:+.6f})")

if abs(beta_drift_full) > 0.05:
    flag(f"Beta drifted {beta_drift_full:+.4f} over the full year",
         "Kalman spread will materially differ from static OLS spread")
else:
    print(f"  [  OK  ] Beta stable: total drift = {beta_drift_full:+.4f} (<0.05)")

# ── Section 3: Spread & characterization comparison ───────────────────────────
print("\n--- Section 3: Spread Characterization (Formation Period) ---")

# Slice Kalman betas to formation period
alpha_form_k = alpha_full.loc[formation.index]
beta_form_k  = beta_full.loc[formation.index]

# Kalman formation spread (diagnostic only -- NOT used for window)
# The Kalman posterior spread at bar t uses theta_t (state AFTER observing bar t),
# making it approximately equal to the scaled Kalman innovation — white noise by
# construction.  HL and Hurst here reflect that artifact, not genuine mean-reversion.
spread_f_k   = compute_spread(formation["close_a"], formation["close_b"],
                               alpha_form_k, beta_form_k)
hl_k, _      = compute_half_life(spread_f_k)
H_form_k     = hurst_exponent(spread_f_k)
_, _         = window_from_half_life(hl_k)   # computed but not used (see below)

# Static OLS formation spread
spread_f_ols = compute_spread(formation["close_a"], formation["close_b"],
                               alpha_ols, beta_ols)
hl_ols, _    = compute_half_life(spread_f_ols)
H_form_ols   = hurst_exponent(spread_f_ols)
window_ols, _= window_from_half_life(hl_ols)

# Option A: Kalman window anchored to OLS formation window.
# Rationale: the Kalman posterior formation spread is white noise (HL≈0.6 bars),
# which would force window=10 (minimum) and produce ~1,300 trades at 10.7/day --
# not a pairs-trading signal.  The OLS formation spread correctly captures the
# mean-reversion timescale (HL=679.7 bars → window=240) because it uses a fixed
# beta and therefore retains the autocorrelation structure.
window_k = window_ols

print(f"\n  {'Metric':<25} {'Static OLS':>12} {'Kalman':>12}")
print(f"  {'-'*52}")
print(f"  {'Half-life (bars)':<25} {hl_ols:>12.1f} {hl_k:>12.1f}  [Kalman: artifact -- see note]")
print(f"  {'Hurst (formation)':<25} {H_form_ols:>12.4f} {H_form_k:>12.4f}  [Kalman: near-zero -- artifact]")
print(f"  {'Window (bars)':<25} {window_ols:>12}  {window_k:>10} (OLS anchor)")
print(f"  [NOTE] Kalman window forced = OLS window (Option A). Kalman posterior"
      f" spread HL is not a valid reversion timescale.")

all_passed &= check("Kalman formation HL is a positive finite number",
                    np.isfinite(hl_k) and hl_k > 0,
                    f"HL={hl_k:.1f}")
all_passed &= check("Kalman formation Hurst is finite",
                    np.isfinite(H_form_k), f"H={H_form_k:.4f}")

# ── Section 4: Z-score diagnostics comparison (trading period) ────────────────
print("\n--- Section 4: Z-Score Diagnostics (Trading Period) ---")

alpha_trade_k = alpha_full.loc[trading.index]
beta_trade_k  = beta_full.loc[trading.index]

spread_t_k   = compute_spread(trading["close_a"], trading["close_b"],
                               alpha_trade_k, beta_trade_k)
spread_t_ols = compute_spread(trading["close_a"], trading["close_b"],
                               alpha_ols, beta_ols)

zdf_k   = compute_zscore(spread_t_k,   window=window_k)
zdf_ols = compute_zscore(spread_t_ols, window=window_ols)

z_k   = zdf_k["zscore"].dropna()
z_ols = zdf_ols["zscore"].dropna()

hl_t_k,   _ = compute_half_life(spread_t_k)
hl_t_ols, _ = compute_half_life(spread_t_ols)
H_t_k       = hurst_exponent(spread_t_k)
H_t_ols     = hurst_exponent(spread_t_ols)

cov_k   = empirical_coverage(z_k)
cov_ols = empirical_coverage(z_ols)
gap_k   = float(cov_k.loc[cov_k["threshold"] == "+-2.0", "gap_pct"].values[0])
gap_ols = float(cov_ols.loc[cov_ols["threshold"] == "+-2.0", "gap_pct"].values[0])

tb_k   = tail_behavior(z_k)
tb_ols = tail_behavior(z_ols)

pos_k   = generate_positions(zdf_k["zscore"],   entry_z=2.0, exit_z=0.0)
pos_ols = generate_positions(zdf_ols["zscore"], entry_z=2.0, exit_z=0.0)
n_k     = count_trades(pos_k)
n_ols   = count_trades(pos_ols)

xval_k   = quantile_crossval(compute_zscore(spread_f_k,   window=window_k)["zscore"].dropna(), z_k)
xval_ols = quantile_crossval(compute_zscore(spread_f_ols, window=window_ols)["zscore"].dropna(), z_ols)

# Rolling Hurst % windows
def pct_mr_windows(spread, rolling_window=5000, step=500):
    vals = spread.dropna()
    hs = []
    for start in range(0, len(vals) - rolling_window, step):
        H = hurst_exponent(vals.iloc[start:start + rolling_window], max_lag=100)
        hs.append(H)
    return (np.array(hs) < 0.5).mean() * 100 if hs else float("nan")

pct_mr_k   = pct_mr_windows(spread_t_k)
pct_mr_ols = pct_mr_windows(spread_t_ols)

print(f"\n  {'Metric':<30} {'Static OLS':>12} {'Kalman':>12}  {'Change':>10}")
print(f"  {'-'*70}")

rows = [
    ("HL trading (bars)",          f"{hl_t_ols:12.1f}", f"{hl_t_k:12.1f}", f"{hl_t_k-hl_t_ols:+10.1f}"),
    ("Hurst trading",              f"{H_t_ols:12.4f}", f"{H_t_k:12.4f}",   f"{H_t_k-H_t_ols:+10.4f}"),
    ("Z-score std",                f"{tb_ols['std']:12.4f}", f"{tb_k['std']:12.4f}", f"{tb_k['std']-tb_ols['std']:+10.4f}"),
    ("Excess kurtosis",            f"{tb_ols['excess_kurtosis']:12.4f}", f"{tb_k['excess_kurtosis']:12.4f}", f"{tb_k['excess_kurtosis']-tb_ols['excess_kurtosis']:+10.4f}"),
    ("Coverage gap +-2s (%)",      f"{gap_ols:12.2f}", f"{gap_k:12.2f}",   f"{gap_k-gap_ols:+10.2f}"),
    ("Regime shift (drift)",       f"{xval_ols['abs_drift']:12.4f}", f"{xval_k['abs_drift']:12.4f}", ""),
    ("Trades (fixed 2.0)",         f"{n_ols:>12}", f"{n_k:>12}",           f"{n_k-n_ols:>+10}"),
    ("% Rolling windows H<0.5",    f"{pct_mr_ols:12.1f}", f"{pct_mr_k:12.1f}", f"{pct_mr_k-pct_mr_ols:+10.1f}"),
]

for label, v_ols, v_k, change in rows:
    print(f"  {label:<30} {v_ols}  {v_k}  {change}")

# Key checks
all_passed &= check("Kalman z_std is finite and positive",
                    np.isfinite(tb_k["std"]) and tb_k["std"] > 0,
                    f"{tb_k['std']:.4f}")

if tb_k["std"] < tb_ols["std"]:
    print(f"\n  [  OK  ] Kalman Z-score std ({tb_k['std']:.4f}) < Static ({tb_ols['std']:.4f})"
          f" -- window-truncation bias reduced")
else:
    flag(f"Kalman std ({tb_k['std']:.4f}) >= Static std ({tb_ols['std']:.4f})",
         "Kalman spread may be more dispersed -- check beta stability")

if gap_k < gap_ols:
    print(f"  [  OK  ] Kalman coverage gap ({gap_k:.2f}%) < Static ({gap_ols:.2f}%)"
          f" -- distribution closer to normal")
else:
    flag(f"Kalman coverage gap ({gap_k:.2f}%) >= Static ({gap_ols:.2f}%)",
         "Kalman did not reduce fat-tail artifact in this dataset")

# ── Section 5: Beta drift plot ────────────────────────────────────────────────
print("\n--- Section 5: Plots ---")

os.makedirs(FIG_DIR, exist_ok=True)
formation_end_ts = formation.index[-1]

try:
    plot_kalman_beta(
        beta_series=beta_full,
        ticker_a=ta,
        ticker_b=tb,
        formation_end=formation_end_ts,
        static_beta=beta_ols,
        save_path=os.path.join(FIG_DIR, f"kalman_beta_{ta}_{tb}.png"),
    )
    all_passed &= check(f"kalman_beta_{ta}_{tb}.png saved",
                        os.path.exists(os.path.join(FIG_DIR, f"kalman_beta_{ta}_{tb}.png")))
except Exception as e:
    print(f"  [{FAIL}] Beta drift plot failed: {e}")
    all_passed = False

# ── Section 6: Signal CSV output ─────────────────────────────────────────────
print("\n--- Section 6: Signal CSV ---")

from src.pipeline.run_week2 import run_pipeline

result_k = run_pipeline(
    trading["close_a"], trading["close_b"],
    alpha=alpha_trade_k, beta=beta_trade_k,
    window=window_k, entry_z=2.0, exit_z=0.0,
)

os.makedirs(SIGNALS_DIR, exist_ok=True)
csv_path = os.path.join(SIGNALS_DIR, f"signals_{ta}_{tb}_kalman.csv")
result_k.to_csv(csv_path)

all_passed &= check("Kalman signal CSV saved", os.path.exists(csv_path), csv_path)
all_passed &= check("Kalman CSV has correct columns",
                    {"spread","rolling_mean","rolling_std","zscore",
                     "position","signal_valid","position_executed"}.issubset(result_k.columns))
all_passed &= check("Kalman CSV > 40000 rows", len(result_k) > 40_000,
                    f"{len(result_k):,} rows")

# ── Final ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Kalman validation complete.\033[0m")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
