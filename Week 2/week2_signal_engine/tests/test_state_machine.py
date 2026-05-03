"""
Unit tests for Position Logic State Machine

Each test constructs a minimal hand-crafted Z-score sequence and verifies
the exact position array.  This makes the expected transitions explicit and
easy to audit.

State machine rules (entry_z=2.0, exit_z=0.0):
  flat  -> long  : z < -2.0
  flat  -> short : z >  2.0
  long  -> flat  : z >= 0.0
  short -> flat  : z <= 0.0
  NaN             : hold current state unchanged
"""
import unittest
import numpy as np
import pandas as pd

from src.signals.state_machine import generate_positions, count_trades


class TestSingleExcursion(unittest.TestCase):

    def test_long_excursion(self):
        """Single dive below -entry_z returns to flat after crossing zero."""
        z   = pd.Series([0.0, 0.0, -2.5, -2.5, 0.5])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        exp = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp,
            err_msg="Long excursion: expected [0,0,1,1,0]")

    def test_short_excursion(self):
        """Single spike above +entry_z returns to flat after crossing zero."""
        z   = pd.Series([0.0, 2.5, 0.5, -0.1])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        # Bar 1: z=2.5 > 2.0 -> short (-1)
        # Bar 2: z=0.5 > 0   -> short still (exits only when z <= 0)
        # Bar 3: z=-0.1 <= 0 -> flat
        exp = np.array([0, -1, -1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp,
            err_msg="Short excursion: expected [0,-1,-1,0]")

    def test_single_excursion_is_one_trade(self):
        """count_trades must return 1 for a single long excursion."""
        z   = pd.Series([0.0, -2.5, -2.5, 0.5, 0.0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        self.assertEqual(count_trades(pos), 1)


class TestExitBeforeReentry(unittest.TestCase):

    def test_no_reentry_without_zero_crossing(self):
        """Z dips deeper twice without crossing zero — only ONE entry."""
        z   = pd.Series([0.0, -2.5, -3.0, -2.1, 0.5])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        exp = np.array([0, 1, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp)
        self.assertEqual(count_trades(pos), 1,
            msg="Repeated dips below entry_z without crossing 0 must count as 1 trade")

    def test_two_trades_after_zero_crossing(self):
        """Two separate excursions separated by a zero-crossing = 2 trades."""
        z   = pd.Series([0.0, -2.5, 0.5, 0.0, -2.5, 0.5])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        # Trade 1: enter bar 1, exit bar 2
        # Trade 2: enter bar 4, exit bar 5
        exp = np.array([0, 1, 0, 0, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp)
        self.assertEqual(count_trades(pos), 2)

    def test_direction_flip_requires_zero_crossing(self):
        """Cannot go directly from long to short without passing through flat."""
        # With exit_z=0.0: long exits when z >= 0; short enters when z > 2.
        # Z sequence designed to: enter long, cross zero, enter short
        z   = pd.Series([0.0, -2.5, 0.1, 2.5, 0.0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        # Bar 0: flat
        # Bar 1: z=-2.5 -> long (+1)
        # Bar 2: z=0.1 >= 0 -> flat (0)
        # Bar 3: z=2.5 > 2.0 -> short (-1)
        # Bar 4: z=0.0 <= 0 -> flat (0)
        exp = np.array([0, 1, 0, -1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp)
        self.assertEqual(count_trades(pos), 2)


class TestNaNHandling(unittest.TestCase):

    def test_nan_holds_flat_state(self):
        """NaN while flat stays flat."""
        z   = pd.Series([0.0, float("nan"), float("nan"), 0.0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        exp = np.array([0, 0, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp)

    def test_nan_holds_long_state(self):
        """NaN bars during a long position maintain the +1 state."""
        z   = pd.Series([0.0, -2.5, float("nan"), float("nan"), 0.5])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        exp = np.array([0, 1, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp,
            err_msg="NaN bars inside position must hold state, not trigger exit")

    def test_nan_holds_short_state(self):
        """NaN bars during a short position maintain the -1 state."""
        z   = pd.Series([0.0, 2.5, float("nan"), -0.1])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        exp = np.array([0, -1, -1, 0], dtype=np.int8)
        np.testing.assert_array_equal(pos.values, exp)

    def test_leading_nan_does_not_enter(self):
        """Leading NaN bars (burn-in) must not trigger any entry."""
        z   = pd.Series([float("nan")] * 5 + [-2.5, 0.5])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        # NaN bars stay flat; entry only at bar 5
        self.assertEqual(int(pos.iloc[4]), 0,
            msg="NaN burn-in bar must remain flat")
        self.assertEqual(int(pos.iloc[5]), 1,
            msg="Entry must fire after burn-in ends")


class TestCountTrades(unittest.TestCase):

    def test_count_no_trades(self):
        z   = pd.Series([0.0, 0.0, 0.0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        self.assertEqual(count_trades(pos), 0)

    def test_count_many_trades(self):
        # Alternating: flat / enter long / exit / flat / enter short / exit
        z_vals = []
        for _ in range(10):
            z_vals += [0.0, -2.5, 0.5]   # long excursion
            z_vals += [0.0,  2.5, -0.1]  # short excursion
        z   = pd.Series(z_vals)
        pos = generate_positions(z, entry_z=2.0, exit_z=0.0)
        self.assertEqual(count_trades(pos), 20)


if __name__ == "__main__":
    unittest.main()
