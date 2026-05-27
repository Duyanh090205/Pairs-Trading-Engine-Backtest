"""Smoke: alpha refit on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))

from engine_daily.alpha_refit import recompute_alpha


def test_basic():
    """Stationary spread around alpha_true=2.5 -> recompute should recover it."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.bdate_range("2022-01-03", periods=n)
    # resid_b ~ random walk; resid_a = alpha_true + beta*resid_b + noise
    rb = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    beta_true = 1.5
    alpha_true = 2.5
    noise = pd.Series(rng.normal(0, 0.02, n), index=idx)
    ra = alpha_true + beta_true * rb + noise

    alpha_hat = recompute_alpha(ra, rb, beta_true, n_lookback=60)
    assert abs(alpha_hat - alpha_true) < 0.05, \
        f"alpha_hat={alpha_hat:.4f} too far from alpha_true={alpha_true:.4f}"
    print(f"[PASS] basic: alpha_hat={alpha_hat:.4f} ~ alpha_true={alpha_true:.4f}")


def test_uses_only_tail():
    """If the EARLY part of formation has different alpha, refit on tail should reflect tail."""
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.bdate_range("2022-01-03", periods=n)
    rb = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    beta = 1.0
    # First 140 bars: alpha=0; last 60: alpha=3
    alpha_first = 0.0
    alpha_last = 3.0
    noise = pd.Series(rng.normal(0, 0.01, n), index=idx)
    ra_vals = beta * rb.values + noise.values
    ra_vals[:140] += alpha_first
    ra_vals[140:] += alpha_last
    ra = pd.Series(ra_vals, index=idx)

    alpha_hat = recompute_alpha(ra, rb, beta, n_lookback=60)
    assert abs(alpha_hat - alpha_last) < 0.1, \
        f"alpha_hat={alpha_hat:.4f} should be near alpha_last={alpha_last:.4f}"
    print(f"[PASS] uses_only_tail: alpha_hat={alpha_hat:.4f} (tail), "
          f"not avg of full ({(alpha_first*140+alpha_last*60)/200:.4f})")


def test_short_series():
    """Fewer than n_lookback bars: should use all available, warn."""
    rng = np.random.default_rng(2)
    n = 30
    idx = pd.bdate_range("2022-01-03", periods=n)
    rb = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    ra = 1.0 + 1.0 * rb + pd.Series(rng.normal(0, 0.01, n), index=idx)
    alpha = recompute_alpha(ra, rb, 1.0, n_lookback=60)
    assert abs(alpha - 1.0) < 0.1, f"alpha={alpha:.4f}"
    print(f"[PASS] short_series: alpha={alpha:.4f} on n=30 with n_lookback=60")


def test_bad_inputs():
    """Invalid beta or n_lookback should raise."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    ra = pd.Series(np.zeros(100), index=idx)
    rb = pd.Series(np.zeros(100), index=idx)
    try:
        recompute_alpha(ra, rb, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Should have raised on beta<0")
    try:
        recompute_alpha(ra, rb, 1.0, n_lookback=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Should have raised on n_lookback<10")
    print(f"[PASS] bad_inputs: raised on beta<0 and n_lookback<10")


if __name__ == "__main__":
    test_basic()
    test_uses_only_tail()
    test_short_series()
    test_bad_inputs()
    print("\ntest_alpha_refit: ALL PASSED")
