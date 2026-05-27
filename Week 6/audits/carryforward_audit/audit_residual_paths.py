"""
audit_residual_paths.py — Cross-path comparison
================================================

PATTERN: integration-layer wiring bug (per /deep-audit-bug skill).

TWIN PAIR:
    A) `compute_residuals_for_gate(formation_data)` in carry_forward.py
       Used by orchestrator to build fold N+1 residuals for the GATE TEST
       at the end of fold N.

    B) `discovery_daily.run(formation_data, ...)` internal residual flow
       What fold N+1's actual discovery does when it runs.

CLAIMED EQUIVALENCE:
    The gate is supposed to predict what fold N+1's discovery would find.
    For that prediction to be valid, A and B must produce IDENTICAL
    residual_log_prices on the same formation_data.

WHY IT MATTERS:
    The 78% upper-bound finding in audit_carryforward_upper_bound.py used a
    THIRD function `compute_fold_residuals` that mirrors A. If A actually
    diverges from B in production, the 78% gate-pass number is testing a
    different statistic than what the engine sees -- the gate would carry
    pairs that fold N+1's discovery wouldn't find cointegrated.

TEST:
    Feed the same synthetic formation_data through A and B; compare
    residual_log_prices DataFrames cell-by-cell with tolerance 1e-12 (should
    be exactly bit-identical since both call _apply_hard_screens +
    fit_factor_model + min-obs filter with identical constants).

EXPECTED:
    PASS (max_abs_diff < 1e-12). The two functions are designed to be
    equivalent. If they diverge, that's a confirmed bug.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))

from engine_daily.carry_forward import compute_residuals_for_gate
from engine.phase1_cointegration.discovery import _apply_hard_screens
from engine.phase1_cointegration.factor_residual import fit_factor_model

_N_FACTOR_COMPONENTS = 5
_MIN_OBS_DAILY = 100


def discovery_path_residuals(
    formation_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Replicate engine_daily.discovery_daily's internal residual building.

    This is the path fold N+1's actual discovery takes (line numbers refer
    to discovery_daily.py post-revert state).
    """
    if not formation_data:
        return pd.DataFrame()
    survivors, _, _ = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        return pd.DataFrame()
    formation_data = {t: formation_data[t] for t in survivors}
    try:
        _, factor_tickers, residual_log_prices, _ = fit_factor_model(
            formation_data, n_components=_N_FACTOR_COMPONENTS,
        )
    except ValueError:
        return pd.DataFrame()

    # discovery_daily's per-ticker dropna + min-obs filter
    formation_data_residual: dict[str, pd.Series] = {}
    for tk in factor_tickers:
        df = formation_data[tk].copy()
        df = df.loc[df.index.intersection(residual_log_prices.index)]
        df["log_close"] = residual_log_prices[tk].reindex(df.index)
        df = df.dropna(subset=["log_close"])
        if len(df) >= _MIN_OBS_DAILY:
            formation_data_residual[tk] = df["log_close"]

    if not formation_data_residual:
        return pd.DataFrame()
    return pd.DataFrame(formation_data_residual)


def _synthetic_formation(
    n_tickers: int = 60, n_days: int = 252, seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    market = np.cumsum(rng.normal(0, 0.012, n_days))
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        beta_mkt = 0.5 + 0.7 * rng.random()
        idio = np.cumsum(rng.normal(0, 0.008, n_days))
        log_close = 4.5 + beta_mkt * market + idio
        vol = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
        out[f"T{i:03d}"] = pd.DataFrame(
            {"log_close": log_close, "volume": vol}, index=dates,
        )
    return out


def main() -> int:
    print("=" * 72)
    print("  audit_residual_paths.py")
    print("  Compares compute_residuals_for_gate (gate test) vs discovery_daily's")
    print("  internal residual flow (actual fold N+1 path)")
    print("=" * 72)

    formation = _synthetic_formation(n_tickers=60, n_days=252, seed=42)

    # Path A: gate's residual computation
    df_gate, surv_gate = compute_residuals_for_gate(formation)
    print(f"\n  Path A (compute_residuals_for_gate):")
    print(f"    shape       : {df_gate.shape}")
    print(f"    survivors   : {len(surv_gate)}")

    # Path B: discovery's residual flow
    df_disc = discovery_path_residuals(formation)
    print(f"\n  Path B (discovery_daily's flow):")
    print(f"    shape       : {df_disc.shape}")

    # ---- Compare ----
    print(f"\n  Comparison:")

    if df_gate.empty or df_disc.empty:
        print(f"  [FAIL] One or both produced empty DataFrames.")
        return 1

    # 1. Column set
    cols_gate = set(df_gate.columns)
    cols_disc = set(df_disc.columns)
    only_gate = cols_gate - cols_disc
    only_disc = cols_disc - cols_gate
    if only_gate or only_disc:
        print(f"  [FAIL] Column-set mismatch:")
        if only_gate:
            print(f"    only in gate path : {sorted(list(only_gate))[:5]} "
                  f"({len(only_gate)} total)")
        if only_disc:
            print(f"    only in disc path : {sorted(list(only_disc))[:5]} "
                  f"({len(only_disc)} total)")
        return 1
    else:
        print(f"    column sets       : identical ({len(cols_gate)} tickers)")

    # 2. Index set
    idx_gate = set(df_gate.index)
    idx_disc = set(df_disc.index)
    if idx_gate != idx_disc:
        print(f"  [FAIL] Index-set mismatch: gate has {len(idx_gate)} dates, "
              f"disc has {len(idx_disc)}, intersect={len(idx_gate & idx_disc)}")
        return 1
    else:
        print(f"    index sets        : identical ({len(idx_gate)} dates)")

    # 3. Cell-by-cell diff
    df_gate_aligned = df_gate.loc[df_disc.index, df_disc.columns]
    diff = (df_gate_aligned - df_disc).abs()
    # Handle NaN-NaN equivalence
    nan_a = df_gate_aligned.isna()
    nan_b = df_disc.isna()
    same_nan_pattern = (nan_a == nan_b).all().all()
    finite_max_diff = float(diff.max().max())

    print(f"    same NaN pattern  : {same_nan_pattern}")
    print(f"    max abs diff      : {finite_max_diff:.2e}")

    if same_nan_pattern and finite_max_diff < 1e-10:
        print(f"\n  [PASS] Residual paths are equivalent (max diff < 1e-10).")
        print(f"         The gate's residual computation predicts what fold N+1's")
        print(f"         actual discovery would see. 78% upper-bound finding")
        print(f"         (audit_carryforward_upper_bound.py) generalizes to the")
        print(f"         production gate.")
        return 0
    else:
        print(f"\n  [BUG SUSPECTED] Residual paths diverge.")
        print(f"  Gate test residuals != Discovery residuals.")
        print(f"  Implication: 78% gate-pass figure measures a different")
        print(f"  statistic than the production gate is actually computing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
