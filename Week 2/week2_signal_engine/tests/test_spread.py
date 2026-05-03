"""
Unit tests for Spread Calculation and OLS Hedge Ratio
"""
import unittest
import numpy as np
import pandas as pd

from src.signals.spread import estimate_hedge_ratio, compute_spread


class TestHedgeRatio(unittest.TestCase):

    def test_hedge_ratio_recovery(self):
        """OLS should recover known alpha/beta from synthetic data."""
        rng    = np.random.default_rng(0)
        log_b  = rng.standard_normal(2000)
        log_a  = 1.5 + 0.8 * log_b + rng.normal(0, 0.005, 2000)
        a      = pd.Series(np.exp(log_a))
        b      = pd.Series(np.exp(log_b))

        alpha, beta = estimate_hedge_ratio(a, b)

        self.assertAlmostEqual(alpha, 1.5, delta=0.02,
            msg="Alpha (intercept) not recovered within tolerance")
        self.assertAlmostEqual(beta, 0.8, delta=0.005,
            msg="Beta (hedge ratio) not recovered within tolerance")

    def test_hedge_ratio_exact_fit(self):
        """Zero-noise data must recover alpha and beta exactly."""
        log_b  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        log_a  = 0.5 + 1.2 * log_b           # exact linear relationship
        a      = pd.Series(np.exp(log_a))
        b      = pd.Series(np.exp(log_b))

        alpha, beta = estimate_hedge_ratio(a, b)

        self.assertAlmostEqual(alpha, 0.5, places=8)
        self.assertAlmostEqual(beta,  1.2, places=8)

    def test_hedge_ratio_returns_floats(self):
        """Return types must be Python floats, not numpy scalars."""
        a = pd.Series([100.0, 101.0, 99.0])
        b = pd.Series([50.0,  51.0,  49.0])
        alpha, beta = estimate_hedge_ratio(a, b)
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta,  float)


class TestComputeSpread(unittest.TestCase):

    def test_spread_zero_when_identical(self):
        """With alpha=0 and beta=1, spread of a series against itself is 0."""
        prices = pd.Series([100.0, 101.0, 99.0, 100.5, 98.0])
        spread = compute_spread(prices, prices, alpha=0.0, beta=1.0)
        np.testing.assert_array_almost_equal(
            spread.values, np.zeros(len(prices)), decimal=10,
            err_msg="Spread of identical series must be identically zero"
        )

    def test_spread_formula(self):
        """Manually verify spread = log(A) - alpha - beta * log(B)."""
        a      = pd.Series([np.e, np.e**2])   # log(a) = [1, 2]
        b      = pd.Series([np.e, np.e**2])   # log(b) = [1, 2]
        alpha  = 0.5
        beta   = 0.5
        # expected: [1 - 0.5 - 0.5*1, 2 - 0.5 - 0.5*2] = [0.0, 0.5]
        spread = compute_spread(a, b, alpha=alpha, beta=beta)
        np.testing.assert_array_almost_equal(spread.values, [0.0, 0.5], decimal=10)

    def test_spread_preserves_index(self):
        """Spread index must match close_a's index."""
        idx = pd.date_range("2022-01-03 09:30", periods=5, freq="1min", tz="UTC")
        a   = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        b   = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        s   = compute_spread(a, b, alpha=0.0, beta=1.0)
        self.assertEqual(list(s.index), list(idx))

    def test_spread_length(self):
        """Spread must have the same length as the input series."""
        a = pd.Series(np.random.default_rng(1).random(300) + 50)
        b = pd.Series(np.random.default_rng(2).random(300) + 50)
        alpha, beta = estimate_hedge_ratio(a, b)
        spread = compute_spread(a, b, alpha, beta)
        self.assertEqual(len(spread), len(a))

    def test_spread_formation_near_zero_mean(self):
        """Spread computed on the same data used to estimate the hedge ratio
        should have a mean close to zero (by OLS construction)."""
        rng    = np.random.default_rng(7)
        log_b  = rng.standard_normal(1000)
        log_a  = 0.3 + 0.9 * log_b + rng.normal(0, 0.02, 1000)
        a      = pd.Series(np.exp(log_a))
        b      = pd.Series(np.exp(log_b))

        alpha, beta = estimate_hedge_ratio(a, b)
        spread      = compute_spread(a, b, alpha, beta)

        self.assertAlmostEqual(float(spread.mean()), 0.0, delta=1e-8,
            msg="Formation-period spread mean must be ~0 by OLS construction")


if __name__ == "__main__":
    unittest.main()
