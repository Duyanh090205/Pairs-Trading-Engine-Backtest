"""
Unit tests for Vectorized Z-Score Engine

Tests cover:
  - NaN burn-in count (min_periods)
  - ddof=1 standard deviation (verified analytically)
  - eps guard against zero-std division
  - No lookahead contamination
  - Output column schema
"""
import unittest
import numpy as np
import pandas as pd

from src.signals.zscore import compute_zscore


class TestZScoreSchema(unittest.TestCase):

    def test_output_columns(self):
        """Result must have exactly rolling_mean, rolling_std, zscore."""
        vals   = pd.Series(np.random.default_rng(0).standard_normal(100))
        result = compute_zscore(vals, window=20)
        for col in ["rolling_mean", "rolling_std", "zscore"]:
            self.assertIn(col, result.columns,
                msg=f"Column '{col}' missing from compute_zscore output")

    def test_index_preserved(self):
        """Output index must match input index exactly."""
        idx    = pd.date_range("2022-01-03 09:30", periods=50, freq="1min", tz="UTC")
        vals   = pd.Series(np.ones(50), index=idx)
        result = compute_zscore(vals, window=10)
        pd.testing.assert_index_equal(result.index, idx)

    def test_output_length(self):
        """Output must have the same number of rows as the input."""
        vals   = pd.Series(np.random.default_rng(1).standard_normal(200))
        result = compute_zscore(vals, window=30)
        self.assertEqual(len(result), len(vals))


class TestZScoreBurnIn(unittest.TestCase):

    def test_nan_count_respects_min_periods(self):
        """There must be at least (min_periods - 1) leading NaNs."""
        vals       = pd.Series(np.random.default_rng(2).standard_normal(500))
        window     = 50
        min_p      = window // 2   # default in compute_zscore
        result     = compute_zscore(vals, window=window)
        n_nan      = int(result["zscore"].isna().sum())
        self.assertGreaterEqual(n_nan, min_p - 1,
            msg=f"Expected >= {min_p - 1} leading NaNs, got {n_nan}")

    def test_explicit_min_periods(self):
        """With min_periods=window, first valid z-score is at index window-1."""
        n      = 100
        window = 20
        vals   = pd.Series(np.random.default_rng(3).standard_normal(n))
        result = compute_zscore(vals, window=window, min_periods=window)
        n_nan  = int(result["zscore"].isna().sum())
        self.assertEqual(n_nan, window - 1,
            msg=f"With min_periods=window, expected exactly {window - 1} NaNs")


class TestDdof(unittest.TestCase):

    def test_ddof_one(self):
        """Verify that rolling std uses ddof=1.

        Window of 2 on [1, 3]:
          mean = 2.0
          std (ddof=1) = sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2)
          z = (3 - 2) / sqrt(2) = 1/sqrt(2)
        """
        vals   = pd.Series([1.0, 3.0, 1.0, 3.0])
        result = compute_zscore(vals, window=2, min_periods=2)

        expected_z = 1.0 / np.sqrt(2)
        actual_z   = float(result["zscore"].iloc[1])
        self.assertAlmostEqual(actual_z, expected_z, places=10,
            msg="Z-score at window boundary must use ddof=1")


class TestEpsGuard(unittest.TestCase):

    def test_constant_series_no_division_by_zero(self):
        """A constant series has std=0. The eps guard must prevent inf/NaN."""
        vals   = pd.Series([5.0] * 200)
        result = compute_zscore(vals, window=20, min_periods=20)
        valid  = result["zscore"].dropna()

        self.assertFalse(valid.isna().any(),
            msg="Z-score must not be NaN when std=0 (eps guard required)")
        self.assertFalse(np.isinf(valid.values).any(),
            msg="Z-score must not be inf when std=0 (eps guard required)")
        # Numerator is 0, denominator is eps -> z ~ 0
        np.testing.assert_array_almost_equal(valid.values, np.zeros(len(valid)),
            decimal=5,
            err_msg="Constant-series z-scores must be near 0 (0/eps)")


class TestNoLookahead(unittest.TestCase):

    def test_future_bar_does_not_affect_past_scores(self):
        """Appending a new bar must not change any previously computed z-scores.

        This verifies that the rolling window is strictly backward-looking.
        If there were any lookahead, the z-scores of bars 0..n-1 would differ
        between the two calls.
        """
        rng        = np.random.default_rng(42)
        vals_short = pd.Series(rng.standard_normal(100))
        # Append a large spike — would contaminate future bars if lookahead
        vals_long  = pd.concat([vals_short, pd.Series([999.0])], ignore_index=True)

        window     = 20
        res_short  = compute_zscore(vals_short, window=window)
        res_long   = compute_zscore(vals_long,  window=window)

        pd.testing.assert_series_equal(
            res_short["zscore"],
            res_long["zscore"].iloc[:100],
            check_names=False,
            check_index=False,
        )

    def test_rolling_uses_only_past_values(self):
        """Z-score at bar t must equal the manually computed z-score using
        only bars [t-window+1, t].
        """
        rng    = np.random.default_rng(99)
        vals   = pd.Series(rng.standard_normal(80))
        window = 20
        result = compute_zscore(vals, window=window, min_periods=window)

        # Check bar 40 manually
        t         = 40
        window_sl = vals.iloc[t - window + 1 : t + 1]
        mu        = float(window_sl.mean())
        sigma     = float(window_sl.std(ddof=1))
        z_manual  = (vals.iloc[t] - mu) / sigma

        self.assertAlmostEqual(
            float(result["zscore"].iloc[t]), z_manual, places=10,
            msg="Z-score at t must equal manual calculation over [t-window+1, t]"
        )


if __name__ == "__main__":
    unittest.main()
