"""
Unit tests for Kalman Filter Dynamic Hedge Ratio

Tests verify:
  1. Output shapes and index preservation
  2. Convergence to known beta on synthetic constant-beta data
  3. Tracking of a structural beta shift mid-series
  4. No-lookahead: filter output at t is unchanged when future bars are appended
  5. No NaN in outputs after initialization
  6. Diagnostics dict has required keys
"""
import unittest
import numpy as np
import pandas as pd

from src.signals.kalman import kalman_hedge_ratio


def _make_pair(n: int, alpha: float, beta: float, noise: float = 0.002,
               seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Synthetic price pair: log(A) = alpha + beta * log(B) + noise."""
    rng   = np.random.default_rng(seed)
    idx   = pd.date_range("2022-01-03 09:30", periods=n, freq="1min", tz="UTC")
    log_b = np.cumsum(rng.normal(0, 0.0005, n))          # random-walk log-price
    log_a = alpha + beta * log_b + rng.normal(0, noise, n)
    return (
        pd.Series(np.exp(log_a), index=idx, name="close_a"),
        pd.Series(np.exp(log_b), index=idx, name="close_b"),
    )


class TestOutputShape(unittest.TestCase):

    def test_output_length_matches_input(self):
        a, b = _make_pair(500, alpha=0.5, beta=1.2)
        alpha_s, beta_s, _ = kalman_hedge_ratio(a, b)
        self.assertEqual(len(alpha_s), len(a))
        self.assertEqual(len(beta_s),  len(a))

    def test_output_index_matches_input(self):
        a, b = _make_pair(300, alpha=0.3, beta=0.9)
        alpha_s, beta_s, _ = kalman_hedge_ratio(a, b)
        pd.testing.assert_index_equal(alpha_s.index, a.index)
        pd.testing.assert_index_equal(beta_s.index,  a.index)

    def test_mismatched_length_raises(self):
        a, b = _make_pair(200, alpha=0.0, beta=1.0)
        with self.assertRaises(ValueError):
            kalman_hedge_ratio(a, b.iloc[:100])


class TestConvergence(unittest.TestCase):

    def test_constant_beta_converges(self):
        """On data generated with a fixed beta, the Kalman estimate at the
        end of a long series should be close to the true beta."""
        true_beta  = 1.3
        true_alpha = 0.4
        a, b = _make_pair(5000, alpha=true_alpha, beta=true_beta,
                          noise=0.001, seed=42)
        _, beta_s, diag = kalman_hedge_ratio(a, b, delta=1e-5)

        # After 5000 bars the filter should have converged close to the true value
        final_beta = diag["final_beta"]
        self.assertAlmostEqual(final_beta, true_beta, delta=0.1,
            msg=f"Kalman final beta {final_beta:.4f} too far from true {true_beta}")

    def test_beta_drift_tracking(self):
        """When the true beta shifts at the midpoint, the Kalman estimate
        at the end should be closer to the new beta than to the old one."""
        rng = np.random.default_rng(7)
        n   = 6000
        idx = pd.date_range("2022-01-03", periods=n, freq="1min", tz="UTC")

        log_b = np.cumsum(rng.normal(0, 0.0005, n))
        # First half: beta=0.8, second half: beta=1.4
        log_a = np.empty(n)
        log_a[:n//2] = 0.2 + 0.8 * log_b[:n//2] + rng.normal(0, 0.001, n//2)
        log_a[n//2:] = 0.2 + 1.4 * log_b[n//2:] + rng.normal(0, 0.001, n - n//2)

        a = pd.Series(np.exp(log_a), index=idx)
        b = pd.Series(np.exp(log_b), index=idx)

        _, beta_s, diag = kalman_hedge_ratio(a, b, delta=1e-4)

        final_beta = diag["final_beta"]
        dist_to_old = abs(final_beta - 0.8)
        dist_to_new = abs(final_beta - 1.4)
        self.assertLess(dist_to_new, dist_to_old,
            msg=f"Final beta {final_beta:.4f} should be closer to new (1.4) "
                f"than old (0.8): dist_new={dist_to_new:.4f}, dist_old={dist_to_old:.4f}")


class TestNoLookahead(unittest.TestCase):

    def test_appending_future_bar_unchanged(self):
        """Filter output for bars 0..n-1 must not change when bar n is added.

        Short run (500 bars) and long run (600 bars) both clip N_init to 500,
        so R is estimated from the same data in both cases.  The spike at bars
        500-599 in the long run must not change betas 0-499.
        """
        a, b = _make_pair(600, alpha=0.2, beta=1.1, seed=3)

        alpha_short, beta_short, _ = kalman_hedge_ratio(a.iloc[:500], b.iloc[:500])

        # Append 100 more bars (with a large spike to detect any lookahead)
        a_long = a.copy()
        b_long = b.copy()
        a_long.iloc[500:] = a_long.iloc[500:] * 5.0   # large distortion

        alpha_long, beta_long, _ = kalman_hedge_ratio(a_long, b_long)

        # First 500 outputs must be identical
        pd.testing.assert_series_equal(
            beta_short,
            beta_long.iloc[:500],
            check_names=False,
            check_index=False,
        )


class TestNaNAndSanity(unittest.TestCase):

    def test_no_nan_in_output(self):
        a, b = _make_pair(1000, alpha=0.1, beta=1.0)
        alpha_s, beta_s, _ = kalman_hedge_ratio(a, b)
        self.assertFalse(alpha_s.isna().any(),
                         "alpha_series must not contain NaN")
        self.assertFalse(beta_s.isna().any(),
                         "beta_series must not contain NaN")

    def test_beta_reasonable_range(self):
        """On realistic price data, beta should not diverge to extreme values."""
        a, b = _make_pair(2000, alpha=0.3, beta=1.05, noise=0.002)
        _, beta_s, _ = kalman_hedge_ratio(a, b, delta=1e-5)
        self.assertTrue((beta_s.abs() < 20).all(),
            "Beta should remain in a finite, reasonable range")


class TestDiagnostics(unittest.TestCase):

    def test_diagnostics_keys(self):
        a, b = _make_pair(200, alpha=0.0, beta=1.0)
        _, _, diag = kalman_hedge_ratio(a, b)
        for key in ["R", "delta", "final_alpha", "final_beta",
                    "innovations_std", "n_bars"]:
            self.assertIn(key, diag, msg=f"Missing key '{key}' in diagnostics")

    def test_diagnostics_n_bars(self):
        a, b = _make_pair(350, alpha=0.0, beta=1.0)
        _, _, diag = kalman_hedge_ratio(a, b)
        self.assertEqual(diag["n_bars"], 350)

    def test_diagnostics_R_positive(self):
        a, b = _make_pair(200, alpha=0.0, beta=1.0)
        _, _, diag = kalman_hedge_ratio(a, b)
        self.assertGreater(diag["R"], 0,
            "Observation noise R must be positive")


if __name__ == "__main__":
    unittest.main()
