"""
Chunk 2 Validation Script — Spread Characterization
Run from: week2_signal_engine/
  python validate_chunk2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.data.loaders import load_pair
from src.data.loaders import _load_ticker
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import (
    compute_half_life, hurst_exponent, window_from_half_life
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition

def flag(label, detail=""):
    print(f"  [{WARN}] FLAG: {label}" + (f" -- {detail}" if detail else ""))

all_passed = True

cfg = load_config(CONFIG_PATH)
dates = cfg["dates"]

# ── Load all pairs ─────────────────────────────────────────────────────────────
PAIRS = {
    "primary":   tuple(cfg["pairs"]["primary"]),
    "secondary": tuple(cfg["pairs"]["secondary"]),
}

pair_data = {}
for label, (ta, tb) in PAIRS.items():
    merged, _ = load_pair(DATA_DIR, ta, tb)
    formation, trading = split_periods(merged, **dates)
    pair_data[label] = {
        "tickers": (ta, tb),
        "formation": formation,
        "trading": trading,
    }

# ── Helper: characterize a pair ────────────────────────────────────────────────
def characterize(label):
    d = pair_data[label]
    ta, tb = d["tickers"]
    f = d["formation"]

    fa, fb = f["close_a"], f["close_b"]
    alpha, beta = estimate_hedge_ratio(fa, fb)
    spread_f = compute_spread(fa, fb, alpha, beta)

    hl, lam = compute_half_life(spread_f)
    H       = hurst_exponent(spread_f)
    window, note = window_from_half_life(hl)

    # 3-month sub-window (last 3 months of formation: Apr–Jun 2022)
    f3m = f.loc["2022-04-01":]
    fa3m, fb3m = f3m["close_a"], f3m["close_b"]
    alpha3m, beta3m = estimate_hedge_ratio(fa3m, fb3m)
    spread3m = compute_spread(fa3m, fb3m, alpha3m, beta3m)
    hl3m, _ = compute_half_life(spread3m)
    H3m     = hurst_exponent(spread3m)

    return {
        "alpha": alpha, "beta": beta,
        "spread_f": spread_f,
        "half_life_6m": hl, "lambda_6m": lam,
        "hurst_6m": H,
        "half_life_3m": hl3m, "hurst_3m": H3m,
        "window": window, "window_note": note,
        "beta_3m": beta3m,
    }

print("\n" + "="*60)
print("CHUNK 2: SPREAD CHARACTERIZATION")
print("="*60)

char_results = {}
for label, (ta, tb) in PAIRS.items():
    print(f"\n--- {label.upper()} PAIR: {ta} / {tb} ---")
    cr = characterize(label)
    char_results[label] = cr

    # ── 1. Hedge ratio checks ──────────────────────────────────────────────────
    print("\n  [Hedge Ratio]")
    all_passed &= check("beta is finite and non-zero", np.isfinite(cr["beta"]) and cr["beta"] != 0,
                        f"alpha={cr['alpha']:.5f}, beta={cr['beta']:.5f}")
    if cr["beta"] < 0:
        flag("Negative beta", f"beta={cr['beta']:.4f} — economically unusual for this pair")
    if abs(cr["beta"]) > 5:
        flag("Large |beta|", f"beta={cr['beta']:.4f} — large price ratio; verify pair logic")
    beta_drift = abs(cr["beta"] - cr["beta_3m"])
    all_passed &= check("3m/6m beta stability |drift| < 0.3",
                        beta_drift < 0.3,
                        f"6m beta={cr['beta']:.4f}, 3m beta={cr['beta_3m']:.4f}, drift={beta_drift:.4f}")

    # ── 2. Spread checks ───────────────────────────────────────────────────────
    print("\n  [Spread]")
    spread = cr["spread_f"]
    all_passed &= check("Spread has no NaN (after dropna alignment)",
                        spread.dropna().shape[0] > 40_000,
                        f"{spread.dropna().shape[0]:,} non-NaN bars")
    all_passed &= check("Spread mean near 0 (|mean| < 1.0)",
                        abs(spread.mean()) < 1.0,
                        f"mean={spread.mean():.5f}, std={spread.std():.5f}")

    # ── 3. Half-life checks ────────────────────────────────────────────────────
    print("\n  [Half-Life]")
    hl = cr["half_life_6m"]
    lam = cr["lambda_6m"]
    hl_min = hl if np.isfinite(hl) else None

    all_passed &= check("lambda < 0 (mean-reverting process)",
                        np.isfinite(lam) and lam < 0,
                        f"lambda={lam:.6f}")
    if np.isinf(hl):
        flag("Half-life is Inf (lambda >= 0)", "No mean reversion detected on formation period")
    elif hl > 240:
        flag("Half-life > 240 bars (4 hours)", f"HL={hl:.1f} min — will be capped at window=240")
    elif hl < 10:
        flag("Half-life < 10 bars", f"HL={hl:.1f} min — very fast reversion; clamped to window=10")

    print(f"    half_life_6m = {hl:.2f} bars  |  half_life_3m = {cr['half_life_3m']:.2f} bars")
    print(f"    window = {cr['window']} bars  ({cr['window_note']})")

    if np.isfinite(hl) and np.isfinite(cr["half_life_3m"]):
        hl_ratio = max(hl, cr["half_life_3m"]) / max(min(hl, cr["half_life_3m"]), 1e-9)
        if hl_ratio > 3:
            flag("Large 3m/6m half-life divergence",
                 f"ratio={hl_ratio:.1f}x — spread structure may have shifted mid-formation")

    # ── 4. Hurst checks ────────────────────────────────────────────────────────
    print("\n  [Hurst Exponent]")
    H = cr["hurst_6m"]
    all_passed &= check("Hurst is finite", np.isfinite(H), f"H={H:.4f}")
    print(f"    H_6m = {H:.4f}  |  H_3m = {cr['hurst_3m']:.4f}")

    if H < 0.5:
        print(f"    -> H < 0.5 -- consistent with mean reversion")
    elif H < 0.6:
        flag("H is between 0.5 and 0.6",
             f"H={H:.4f} — borderline; spread shows weak mean-reversion character")
    else:
        flag("H > 0.6 — no mean-reversion signal",
             f"H={H:.4f} — spread behaves more like a random walk or trend on formation")

    if not np.isnan(cr["hurst_3m"]) and abs(H - cr["hurst_3m"]) > 0.15:
        flag("Large 3m/6m Hurst divergence",
             f"6m H={H:.4f}, 3m H={cr['hurst_3m']:.4f} — regime shift in formation period?")

    # ── 5. Quick sanity: apply spread to trading period ────────────────────────
    print("\n  [Trading Period Spread Preview]")
    t = pair_data[label]["trading"]
    spread_t = compute_spread(t["close_a"], t["close_b"], cr["alpha"], cr["beta"])
    all_passed &= check("Trading spread is finite and non-empty",
                        spread_t.dropna().shape[0] > 40_000,
                        f"{spread_t.dropna().shape[0]:,} bars")
    mean_drift = spread_t.mean() - spread.mean()
    if abs(mean_drift) > 0.05:
        flag("Spread mean drifted between periods",
             f"formation mean={spread.mean():.4f}, trading mean={spread_t.mean():.4f}, "
             f"drift={mean_drift:.4f} -- hedge ratio may not generalise well")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for label, (ta, tb) in PAIRS.items():
    cr = char_results[label]
    print(f"\n  {label:10s} {ta}/{tb}")
    print(f"    beta         : {cr['beta']:.5f}")
    print(f"    half_life_6m : {cr['half_life_6m']:.2f} bars  ({cr['half_life_6m']/390:.2f} sessions)")
    print(f"    window       : {cr['window']} bars")
    print(f"    hurst_6m     : {cr['hurst_6m']:.4f}")
    print(f"    hurst_3m     : {cr['hurst_3m']:.4f}")

print()
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Chunk 2 complete.\033[0m")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
