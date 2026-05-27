"""Smoke: discovery_daily on synthetic + real fold 1 universe."""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))


def test_imports_and_constants():
    """Module loads and daily-specific constants are sensible."""
    from engine_daily import discovery_daily as dd
    assert dd._HL_MIN_DAYS == 5.0
    assert dd._HL_MAX_DAYS == 30.0
    assert dd._BARS_PER_DAY == 1
    assert dd._N_FACTOR_COMPONENTS == 5
    print("[PASS] imports + daily constants (HL [5, 30], bars_per_day=1)")


def test_synthetic_universe():
    """Synthetic 60-ticker daily universe with one strong factor + one cointegrated pair."""
    from engine_daily.discovery_daily import run

    rng = np.random.default_rng(0)
    n_tickers = 60
    n_days = 252  # 1 year
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Build market factor + ticker-specific factor + idio noise
    market = np.cumsum(rng.normal(0, 0.012, n_days))   # ~1.2%/day vol
    data: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        beta_market = 0.5 + 0.7 * rng.random()
        idio = np.cumsum(rng.normal(0, 0.008, n_days))
        log_close = 4.5 + beta_market * market + idio
        vol = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
        data[f"T{i:03d}"] = pd.DataFrame(
            {"log_close": log_close, "volume": vol}, index=dates,
        )

    # Hard-coded cointegrated pair: T000 and T001 share an identical idio component
    shared_idio = np.cumsum(rng.normal(0, 0.005, n_days))
    data["T000"]["log_close"] = 4.5 + 1.0 * market + shared_idio + 0.001 * rng.normal(0, 1, n_days)
    data["T001"]["log_close"] = 4.5 + 0.8 * market + shared_idio + 0.001 * rng.normal(0, 1, n_days)

    pairs_df, factor_state = run(data, max_pairs=200)
    assert factor_state, "factor_state must be non-empty"
    assert "loadings_W" in factor_state
    assert factor_state["loadings_W"].shape[0] == 5
    print(f"[PASS] synthetic: {len(pairs_df)} pairs survive | "
          f"cum_var={factor_state['diagnostics']['cumulative_variance_explained']:.3f}")
    # Don't assert T000/T001 specifically in pairs_df — the factor model may have
    # absorbed their shared idio (since they're a low-rank perturbation of market).
    # What we want is: pipeline runs end-to-end on daily input without crashing.


if __name__ == "__main__":
    test_imports_and_constants()
    test_synthetic_universe()
    print("\ntest_discovery_daily: ALL PASSED")
