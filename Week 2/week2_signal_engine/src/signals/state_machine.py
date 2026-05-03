"""
State Machine for Position Tracking

Numba-accelerated path-dependent position logic. Enforces:
  - One entry per excursion (no pyramiding)
  - Mandatory exit through the exit threshold before re-entry
  - NaN Z-scores hold the current position (burn-in / data gaps)
"""
import numpy as np
import pandas as pd
from numba import njit


# ---------------------------------------------------------------------------
# Numba core — operates on raw numpy arrays only
# ---------------------------------------------------------------------------

@njit(cache=True)
def _position_state_machine(
    zscores: np.ndarray,
    entry_z: float,
    exit_z: float,
) -> np.ndarray:
    """Path-dependent position tracker (Numba nopython kernel).

    States
    ------
     0  flat
    +1  long spread  (entered when z < -entry_z, betting on reversion upward)
    -1  short spread (entered when z >  entry_z, betting on reversion downward)

    Transition rules
    ----------------
    flat   -> long   : z < -entry_z
    flat   -> short  : z >  entry_z
    long   -> flat   : z >= -exit_z   (with exit_z=0: when z crosses zero)
    short  -> flat   : z <=  exit_z   (with exit_z=0: when z crosses zero)

    NaN handling
    ------------
    NaN Z-scores (burn-in period, data gaps) leave state unchanged and write
    the current state into positions[i].  This prevents false entry/exit
    signals from empty windows at the start of the trading period.

    !! FLAG — fastmath=True intentionally omitted:
    fastmath enables LLVM's `nnan` flag which assumes inputs are never NaN.
    That breaks the np.isnan(z) guard in this loop, causing NaN bars to
    fall through to comparisons that produce undefined results.  Since the
    NaN-hold behaviour is a correctness requirement (not a nice-to-have),
    fastmath must stay OFF on this function.  The loop is integer-state
    logic; the absence of fastmath has negligible performance impact.

    !! FLAG — position[i] reflects information AT bar i (close price).
    For any PnL calculation this must be shifted by 1 bar before use:
        executed_position = positions.shift(1)
    The shift is Week 3's responsibility; this function outputs position[t]
    as-is, consistent with the execution convention in the research plan.

    Args:
        zscores : float64 numpy array (NaN allowed).
        entry_z : Entry threshold (absolute value).
        exit_z  : Exit threshold (absolute value, typically 0.0).

    Returns:
        int8 numpy array of positions {-1, 0, +1}, same length as zscores.
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    state = 0

    for i in range(n):
        z = zscores[i]

        # Hold state on NaN (burn-in / gap bars)
        if np.isnan(z):
            positions[i] = state
            continue

        if state == 0:
            if z < -entry_z:
                state = 1      # enter long spread
            elif z > entry_z:
                state = -1     # enter short spread
        elif state == 1:       # long spread — exit when z reverts through -exit_z
            if z >= -exit_z:
                state = 0
        elif state == -1:      # short spread — exit when z reverts through exit_z
            if z <= exit_z:
                state = 0

        positions[i] = state

    return positions


def _warmup_state_machine() -> None:
    """JIT-compile _position_state_machine on a tiny dummy array.

    The first call to any @njit function pays compilation overhead (~1–2 s).
    Calling this at import time (or explicitly before the main loop) ensures
    that subsequent calls on real data are fully compiled and fast.
    """
    _position_state_machine(
        np.array([0.0, 2.5, 1.0, -0.1], dtype=np.float64),
        entry_z=2.0,
        exit_z=0.0,
    )


# Trigger compilation at import time so the first real call is not slow.
_warmup_state_machine()


# ---------------------------------------------------------------------------
# Stateless zone classifier (diagnostic — not the trading logic)
# ---------------------------------------------------------------------------

def generate_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.Series:
    """Raw zone classification — stateless, no path-dependence.

    Returns a Series with values:
        +1  long zone   (z < -entry_z)
        -1  short zone  (z >  entry_z)
         0  exit zone   (|z| <= exit_z)
        NaN dead zone   (between exit_z and entry_z, neither entry nor exit)

    !! FLAG — exit_z=0.0 edge case:
        With exit_z=0.0 the "exit zone" condition is z == 0 exactly, which
        never occurs on continuous float data.  In practice ALL values between
        -entry_z and +entry_z will be NaN (dead zone).  This is intentional:
        generate_signals is a diagnostic classifier; the actual exit logic
        lives in _position_state_machine.  Do not confuse these two.

    !! FLAG — signal convention (+1 = long spread, NOT long stock A):
        +1 means "buy the spread" (long A / short B).  This fires when the
        spread is BELOW its mean (z < -entry_z), expecting it to revert up.
        -1 means "sell the spread" (short A / long B).  Fires when z > entry_z.
        This is the mean-reversion convention: trade AGAINST the current
        Z-score direction.
    """
    import numpy as np  # local import keeps module-level deps minimal

    conditions = [
        zscore > entry_z,                                   # short zone
        zscore < -entry_z,                                  # long zone
        (zscore >= -exit_z) & (zscore <= exit_z),          # exit zone
    ]
    values = [-1, 1, 0]
    return pd.Series(
        np.select(conditions, values, default=np.nan),
        index=zscore.index,
        name="raw_signal",
    )


# ---------------------------------------------------------------------------
# Stateful position tracker (pandas wrapper around the Numba kernel)
# ---------------------------------------------------------------------------

def generate_positions(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.Series:
    """Stateful, path-dependent position series.

    Wraps _position_state_machine, handling the pandas <-> numpy boundary.
    The input Z-score Series may contain NaN; those bars hold the current
    state (see _position_state_machine docstring).

    Returns a pd.Series of int8 {-1, 0, +1} with the same index as zscore.
    """
    raw = zscore.to_numpy(dtype=np.float64, na_value=np.nan)
    pos = _position_state_machine(raw, float(entry_z), float(exit_z))
    return pd.Series(pos, index=zscore.index, dtype="int8", name="position")


# ---------------------------------------------------------------------------
# Trade counter
# ---------------------------------------------------------------------------

def count_trades(positions: pd.Series) -> int:
    """Count round-trip trade entries (flat -> non-flat transitions).

    Counts each transition from 0 (flat) to +1 or -1 as one trade entry.
    This gives the number of distinct trades, NOT the number of bars in
    position — a critical distinction for the sensitivity analysis.

    !! FLAG — direction flip not double-counted:
        If the state machine goes directly from +1 to -1 (which cannot
        happen with a 0-exit — the machine always passes through 0 first),
        this function would count it as one entry, not two.  With exit_z=0
        the state always returns to 0 before re-entering, so flip trades
        are impossible.  If exit_z > 0, verify this behaviour still holds.
    """
    prev = positions.shift(1).fillna(0).astype("int8")
    entries = (positions != 0) & (prev == 0)
    return int(entries.sum())
