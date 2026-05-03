# Week 2 Signal Engine — Complete Source Code

All source files from `week2_signal_engine/`, ordered by module.

---


## `configs/params_example.yaml`

```yaml
pairs:
  primary: ["CMS", "DUK"]
  secondary: ["DOW", "LYB"]
  alternatives:
    - ["A", "AFL"]
    - ["AVGO", "GLD"]
    - ["DDOG", "FOXA"]
    - ["HD", "MS"]
    - ["LOW", "MS"]
    - ["DHI", "LVS"]
  benchmarks:
    - ["GOOG", "GOOGL"]
  negative_controls:
    - ["CVNA", "ISRG"]
    - ["INTC", "JPM"]
dates:
  formation_start: "2022-01-01"
  formation_end: "2022-06-30"
  trading_start: "2022-07-01"
  trading_end: "2022-12-31"

engine:
  min_window: 10
  max_window: 2000        # raised from 240: window now equals the half-life (not capped)
  default_window: 60
  entry_z_fixed: 2.0
  exit_z_fixed: 0.0
  use_kalman: false       # true = Kalman dynamic beta; false = static OLS (default)
  kalman_delta: 1.0e-5   # adaptation speed; 1e-5 = slow (~180 trading-day half-life)
  session_open_warmup: 30 # bars to suppress z-score after each session open (0 = disabled)
```

---


## `conftest.py`

```python
"""
pytest conftest — adds the project root to sys.path so tests can import
from src.* without needing an installed package.
"""
import sys
import os

# week2_signal_engine/ is the project root
sys.path.insert(0, os.path.dirname(__file__))
```

---


## `requirements.txt`

```text
pandas
numpy
numba
scipy
matplotlib
pyyaml
```

---


## `src/data/loaders.py`

```python
"""
Data Loaders

Contains logic to iterate raw OHLC CSVs, load selected tickers,
and perform timestamp alignment.
"""
import glob
import os

import pandas as pd

# Regular US equity session in Eastern Time: 9:30 AM – 4:00 PM ET.
# Filter in ET (America/New_York) to handle EST↔EDT transitions correctly.
# In UTC this spans 14:30–21:00 (winter/EST) or 13:30–20:00 (summer/EDT).
_SESSION_OPEN_ET_MIN  = 9 * 60 + 30   # 570 minutes since midnight ET
_SESSION_CLOSE_ET_MIN = 16 * 60        # 960 minutes since midnight ET (exclusive)


def _load_ticker(
    data_dir: str,
    ticker: str,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    """Load all daily CSVs for a single ticker into one sorted, deduplicated DataFrame.

    Args:
        data_dir             : Directory containing TICKER_YYYY-MM-DD.csv files.
        ticker               : Ticker symbol (case-sensitive, matches filenames).
        regular_session_only : If True (default), filter to regular NYSE/NASDAQ
                               session bars (14:30–20:59 UTC, i.e. 9:30–3:59 PM ET).
                               Set False only if you explicitly need extended-hours data.

    Returns a DataFrame with a UTC DatetimeIndex and a single 'close' column.
    Raises FileNotFoundError if no CSV files exist for the ticker.
    Raises ValueError if every file found is empty (or is entirely outside session).
    """
    pattern = os.path.join(data_dir, f"{ticker}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for ticker '{ticker}' in '{data_dir}'. "
            f"Pattern tried: {pattern}"
        )

    frames = []
    for fp in files:
        df = pd.read_csv(fp, usecols=["window_start", "close"])
        if df.empty:
            print(f"WARNING: empty file skipped: {fp}")
            continue
        timestamps = pd.to_datetime(df["window_start"], unit="ns", utc=True)
        df = df.set_index(timestamps)[["close"]]
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No valid data loaded for ticker '{ticker}' — all files were empty."
        )

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="first")]

    if regular_session_only:
        # Convert to Eastern Time so EST↔EDT transitions are handled correctly.
        # NYSE regular session is 9:30–16:00 ET regardless of offset vs UTC.
        eastern_index = combined.index.tz_convert("America/New_York")
        minutes_et = eastern_index.hour * 60 + eastern_index.minute
        mask = (minutes_et >= _SESSION_OPEN_ET_MIN) & (
                minutes_et < _SESSION_CLOSE_ET_MIN)
        combined = combined[mask]

    return combined


def load_pair(
    data_dir: str,
    ticker_a: str,
    ticker_b: str,
    regular_session_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load, align, and inner-join two tickers into a single merged DataFrame.

    Args:
        data_dir             : Directory containing raw CSV files.
        ticker_a / ticker_b  : Ticker symbols.
        regular_session_only : Passed through to _load_ticker (default True).

    Returns:
        merged   : DataFrame with columns ['close_a', 'close_b'], UTC DatetimeIndex.
                   Inner join guarantees zero NaN values.
        audit    : Dict with pre/post join bar counts and alignment rates.
    """
    df_a = _load_ticker(data_dir, ticker_a, regular_session_only=regular_session_only)
    df_b = _load_ticker(data_dir, ticker_b, regular_session_only=regular_session_only)

    bars_a = len(df_a)
    bars_b = len(df_b)

    # Rename before join so columns don't collide
    df_a = df_a.rename(columns={"close": "close_a"})
    df_b = df_b.rename(columns={"close": "close_b"})

    # Inner join on the shared DatetimeIndex — drops any bar missing in either ticker
    merged = df_a.join(df_b, how="inner")

    # Hard invariant: inner join on clean data must produce zero NaNs
    assert not merged.isna().any().any(), (
        "Unexpected NaN values after inner join — check source data for corruption."
    )

    bars_joined = len(merged)
    audit = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "regular_session_only": regular_session_only,
        "bars_a_pre_join": bars_a,
        "bars_b_pre_join": bars_b,
        "bars_after_join": bars_joined,
        "bars_dropped_a": bars_a - bars_joined,
        "bars_dropped_b": bars_b - bars_joined,
        "alignment_rate_a": round(bars_joined / bars_a, 6) if bars_a else 0.0,
        "alignment_rate_b": round(bars_joined / bars_b, 6) if bars_b else 0.0,
        "first_timestamp": merged.index.min(),
        "last_timestamp": merged.index.max(),
    }

    return merged, audit
```

---


## `src/data/validation.py`

```python
"""
Data Validation

Functions for auditing loaded data: checking time gaps, session boundaries,
and calculating coverage relative to expected minute bars.
"""
import pandas as pd


def audit_data(df: pd.DataFrame, ticker: str = "unknown") -> dict:
    """Compute data quality metrics for a single-ticker DataFrame.

    Call this on the output of _load_ticker (pre-join), not the merged pair.
    A post-join audit undercounts gaps because bars unaligned with the partner
    have already been dropped.

    Args:
        df      : DataFrame with a UTC DatetimeIndex and at least one column.
        ticker  : Label used in the returned dict for identification.

    Returns a dict with:
        ticker, total_bars, trading_days, expected_bars, coverage_pct,
        first_timestamp, last_timestamp,
        median_gap, max_gap,
        n_missing_bars, largest_intraday_gap
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"audit_data expects a DatetimeIndex; got {type(df.index).__name__}"
        )
    if df.empty:
        return {
            "ticker": ticker,
            "total_bars": 0,
            "trading_days": 0,
            "expected_bars": 0,
            "coverage_pct": 0.0,
            "first_timestamp": None,
            "last_timestamp": None,
            "median_gap": None,
            "max_gap": None,
            "n_missing_bars": 0,
            "largest_intraday_gap": None,
        }

    assert df.index.is_monotonic_increasing, (
        "Index must be sorted before calling audit_data — run .sort_index() first."
    )

    total_bars = len(df)
    # Unique calendar dates in the index (tz-aware → normalize to midnight UTC)
    trading_days = df.index.normalize().nunique()
    expected_bars = trading_days * 390
    coverage_pct = round(total_bars / expected_bars * 100, 4) if expected_bars else 0.0

    first_timestamp = df.index.min()
    last_timestamp = df.index.max()

    # Gap analysis
    gaps = df.index.to_series().diff().dropna()

    if gaps.empty:
        median_gap = None
        max_gap = None
        n_missing_bars = 0
        largest_intraday_gap = None
    else:
        median_gap = gaps.median()
        max_gap = gaps.max()

        # Intraday gaps: > 1 min but < 6.5 h (the length of a full regular session).
        # With regular_session_only data, the inter-session gap is always >= 17 h
        # (close 21:00 UTC → open 14:30 UTC next day), so 6.5 h cleanly separates
        # genuine within-session holes from overnight/weekend breaks.
        one_min  = pd.Timedelta("1min")
        session_len = pd.Timedelta(hours=6, minutes=30)
        intraday_gaps = gaps[(gaps > one_min) & (gaps < session_len)]
        n_missing_bars = len(intraday_gaps)
        largest_intraday_gap = (
            intraday_gaps.max() if not intraday_gaps.empty else pd.Timedelta(0)
        )

    return {
        "ticker": ticker,
        "total_bars": total_bars,
        "trading_days": trading_days,
        "expected_bars": expected_bars,
        "coverage_pct": coverage_pct,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "median_gap": median_gap,
        "max_gap": max_gap,
        "n_missing_bars": n_missing_bars,
        "largest_intraday_gap": largest_intraday_gap,
    }
```

---


## `src/signals/spread.py`

```python
"""
Spread Construction

OLS hedge ratio estimation (formation period ONLY) and spread computation.
"""
import numpy as np
import pandas as pd


def estimate_hedge_ratio(
    close_a: pd.Series | np.ndarray,
    close_b: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """OLS regression: log(A) = alpha + beta * log(B) + epsilon.

    Must be called on the FORMATION period only.  Calling this on the trading
    period or the full series would introduce lookahead bias.

    Uses np.linalg.lstsq (numerically stable, no scipy dependency).

    Args:
        close_a : Price series for leg A (levels, not log-transformed).
        close_b : Price series for leg B (levels, not log-transformed).

    Returns:
        (alpha, beta) — intercept and slope of the OLS fit.

    !! FLAG: lstsq silently handles near-singular design matrices, but if both
    series have near-zero variance (e.g. a trading halt day slipped through the
    session filter) the returned beta can be numerically garbage.  The caller
    should verify beta > 0 for economically plausible pairs.
    """
    y = np.log(np.asarray(close_a, dtype=np.float64))
    x = np.log(np.asarray(close_b, dtype=np.float64))
    X = np.column_stack([np.ones(len(x)), x])
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(params[0]), float(params[1])
    return alpha, beta


def compute_spread(
    close_a: pd.Series | np.ndarray,
    close_b: pd.Series | np.ndarray,
    alpha: float | pd.Series | np.ndarray,
    beta:  float | pd.Series | np.ndarray,
) -> pd.Series:
    """Spread = log(A) - alpha - beta * log(B).

    The hedge ratio (alpha, beta) must come from the formation period —
    either as a scalar (static OLS) or as a same-length Series / array
    (Kalman filter dynamic hedge ratio).

    When alpha/beta are pd.Series, their pandas index is stripped before
    the arithmetic so that numpy element-wise broadcasting is used rather
    than pandas label alignment (which would produce NaN wherever indices
    differ in type or value).

    Returns a pd.Series preserving the index of close_a if it is a Series,
    otherwise returns a plain Series with a default RangeIndex.
    """
    log_a = np.log(np.asarray(close_a, dtype=np.float64))
    log_b = np.log(np.asarray(close_b, dtype=np.float64))

    # Strip pandas index from alpha/beta if Series — prevents label alignment
    # issues when mixing pd.Series with raw numpy arrays in arithmetic.
    if isinstance(alpha, pd.Series):
        alpha = alpha.values
    if isinstance(beta, pd.Series):
        beta = beta.values

    spread_vals = log_a - alpha - beta * log_b

    if isinstance(close_a, pd.Series):
        return pd.Series(spread_vals, index=close_a.index, name="spread")
    return pd.Series(spread_vals, name="spread")
```

---


## `src/signals/zscore.py`

```python
"""
Z-Score Computation

Vectorized rolling Z-score using pandas Cython accumulators (O(N) time).
"""
import numpy as np
import pandas as pd


def compute_zscore(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """Rolling Z-score of a spread series.

    z(t) = (spread(t) - rolling_mean(t)) / max(rolling_std(t), eps)

    Uses ddof=1 (sample standard deviation) as specified in the research plan.
    The eps guard prevents division-by-zero when the spread is locally constant
    (e.g. a trading halt that slipped through the session filter).

    Args:
        spread      : Spread series produced by compute_spread().
        window      : Look-back window in bars. Derive from half-life via
                      window_from_half_life(); do NOT recalculate on the
                      trading period — that would be lookahead.
        min_periods : Minimum bars required before emitting a Z-score.
                      Defaults to window // 2, so the first valid Z-score
                      appears after half the window has filled.
        eps         : Floor applied to rolling_std before division.
                      Prevents ±Inf Z-scores during low-volatility patches.

    Returns:
        DataFrame with three columns:
            rolling_mean  — rolling window mean of spread
            rolling_std   — rolling window std (ddof=1)
            zscore        — standardised spread value

        The first (min_periods - 1) rows will have NaN in all columns.
        The pipeline's state machine treats NaN Z-scores as "hold current
        position" — see state_machine.py.

    !! FLAG — min_periods choice:
        window // 2 means the Z-score starts emitting halfway through the
        first window.  Those early values are estimated from fewer bars than
        the window implies, so their std is noisier.  For a strict burn-in
        (no signal until the window is fully filled) pass min_periods=window.
        The research plan uses window // 2 for more usable bars at period
        boundaries, which is a reasonable trade-off on 47k-bar datasets.
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = spread.rolling(window, min_periods=min_periods).mean()
    rolling_std  = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    zscore       = (spread - rolling_mean) / rolling_std.clip(lower=eps)

    return pd.DataFrame(
        {
            "rolling_mean": rolling_mean,
            "rolling_std":  rolling_std,
            "zscore":       zscore,
        },
        index=spread.index,
    )


def apply_session_warmup(zdf: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    """Suppress Z-score for the first `warmup_bars` of each calendar session.

    Overnight price gaps cause the first bars of each session to have a spread
    that jumps relative to the prior session's rolling statistics, producing
    artificial Z-score extremes.  Setting those bars to NaN causes the state
    machine to hold its current position (flat at open) rather than entering
    on gap-driven noise.

    Args:
        zdf          : DataFrame from compute_zscore() — must have a DatetimeIndex
                       and a 'zscore' column.
        warmup_bars  : Number of bars to suppress at each session open.
                       30 bars = first 30 minutes of each session.
                       0 = disabled (returns zdf unchanged).

    Returns:
        Copy of zdf with zscore set to NaN for the first `warmup_bars` of each
        unique calendar date.  rolling_mean and rolling_std are left intact so
        the suppression is visible only in the signal, not in the statistics.

    !! FLAG — session detection:
        Sessions are identified by calendar date via index.normalize() (midnight
        of each day in the index timezone).  This is correct for a DST-aware UTC
        index that has already been session-filtered to 09:30-16:00 ET — the
        first bar of each date is always the 09:30 bar.  On early-close days
        (e.g. day before Thanksgiving) the session is shorter, but the first-bar
        detection is unaffected.
    """
    if warmup_bars <= 0:
        return zdf

    df  = zdf.copy()
    z_col = df.columns.get_loc("zscore")

    # Find the integer position of the first bar of each calendar date
    dates      = df.index.normalize()
    unique_dates = np.unique(dates)
    for d in unique_dates:
        locs = np.where(dates == d)[0]
        end  = min(locs[0] + warmup_bars, len(df))
        df.iloc[locs[0]:end, z_col] = np.nan

    return df
```

---


## `src/signals/state_machine.py`

```python
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
```

---


## `src/signals/kalman.py`

```python
"""
Kalman Filter Dynamic Hedge Ratio

Estimates time-varying alpha(t) and beta(t) using a state-space model
where the hedge ratio follows a random walk.  The filter is causal:
beta(t) uses only data from bars 1..t.

State-space model
-----------------
State:       theta_t = [alpha_t, beta_t]
Transition:  theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q)   (random walk)
Observation: log(A_t) = [1, log(B_t)] @ theta_t + v_t,  v_t ~ N(0, R)

Parameters
----------
Q = (delta / (1 - delta)) * I_2    -- process noise; controls adaptation speed
R                                   -- observation noise; estimated from OLS
                                       residuals on the first 500 bars only,
                                       never from the trading period
theta_0 = [0.0, 1.0]               -- neutral prior: zero intercept, unit beta
P_0 = I_2                          -- moderate initial uncertainty

delta guide
-----------
1e-6  very slow   (beta half-life ~1800 trading days -- essentially static)
1e-5  slow        (beta half-life ~180 trading days  -- default; weekly-regime)
1e-4  moderate    (beta half-life ~18 trading days)
1e-3  fast        (beta half-life ~1.8 trading days  -- fits noise)

Usage (in main() / validate_kalman.py)
---------------------------------------
Run on the FULL price series (Jan-Dec). The formation period warms up the
filter; the trading-period beta is sliced afterward.  Never re-initialize
the filter at the formation/trading boundary.

    alpha_full, beta_full, diag = kalman_hedge_ratio(
        merged["close_a"], merged["close_b"], delta=1e-5
    )
    alpha_trade = alpha_full.loc[trading.index]
    beta_trade  = beta_full.loc[trading.index]

!! FLAG -- Python loop performance:
    The Kalman recursion is a Python loop over ~97k bars per pair.  On modern
    hardware this runs in ~0.5-1 second.  Numba is NOT applied here: the loop
    body mixes 2x2 numpy matrix ops with Python control flow, and the compile
    overhead would exceed the runtime for a single pair.  For an 11-pair sweep
    the total cost is ~10 seconds -- acceptable without JIT.

!! FLAG -- R initialization window:
    R is estimated from the first min(500, n//10) bars via OLS.  This window
    must be short relative to the formation period to avoid using too much data
    for prior calibration.  It is deterministic given the data and introduces
    no lookahead because it uses only the earliest available bars.
"""
import numpy as np
import pandas as pd


def kalman_hedge_ratio(
    close_a: pd.Series,
    close_b: pd.Series,
    delta: float = 1e-5,
) -> tuple[pd.Series, pd.Series, dict]:
    """Estimate time-varying hedge ratio via Kalman filter.

    Args:
        close_a : Price series for leg A.  Must share a DatetimeIndex with close_b.
        close_b : Price series for leg B.
        delta   : State noise scalar.  Q = (delta / (1 - delta)) * I_2.
                  Default 1e-5 gives a slow-adapting beta (half-life ~180 trading
                  days) appropriate for utility-pair regime changes.

    Returns:
        alpha_series : pd.Series — time-varying intercept alpha(t), same index as inputs.
        beta_series  : pd.Series — time-varying hedge ratio beta(t), same index as inputs.
        diagnostics  : dict with keys:
                           R               -- observation noise variance used
                           delta           -- adaptation speed parameter
                           final_alpha     -- filter state at last bar
                           final_beta      -- filter state at last bar
                           innovations_std -- std of prediction errors (health check)
                           n_bars          -- total bars processed
    """
    if len(close_a) != len(close_b):
        raise ValueError(
            f"close_a and close_b must have the same length "
            f"({len(close_a)} vs {len(close_b)})"
        )

    n     = len(close_a)
    log_a = np.log(close_a.values.astype(np.float64))
    log_b = np.log(close_b.values.astype(np.float64))

    # ── Process noise covariance ──────────────────────────────────────────────
    Ve = delta / (1.0 - delta)
    Q  = Ve * np.eye(2)

    # ── Observation noise R — estimated from short OLS warm-up window ─────────
    N_init   = min(500, n)
    X_init   = np.column_stack([np.ones(N_init), log_b[:N_init]])
    p_init   = np.linalg.lstsq(X_init, log_a[:N_init], rcond=None)[0]
    resid    = log_a[:N_init] - X_init @ p_init
    R        = max(float(np.var(resid, ddof=1)), 1e-8)   # must be positive

    # ── Initial state and covariance ──────────────────────────────────────────
    theta = np.array([0.0, 1.0])   # neutral prior: alpha=0, beta=1
    P     = np.eye(2)              # moderate uncertainty; converges within ~200 bars

    # ── Storage ───────────────────────────────────────────────────────────────
    alpha_vals  = np.empty(n)
    beta_vals   = np.empty(n)
    innovations = np.empty(n)
    I2          = np.eye(2)

    # ── Kalman recursion ──────────────────────────────────────────────────────
    for i in range(n):
        H = np.array([1.0, log_b[i]])   # observation vector (1x2)

        # Predict
        P_pred = P + Q

        # Innovation
        innov        = log_a[i] - (H @ theta)
        innovations[i] = innov

        # Innovation variance (scalar)
        S = float(H @ P_pred @ H) + R

        # Kalman gain (2,)
        K = (P_pred @ H) / S

        # Update state and covariance
        theta = theta + K * innov
        P     = (I2 - np.outer(K, H)) @ P_pred

        alpha_vals[i] = theta[0]
        beta_vals[i]  = theta[1]

    diagnostics = {
        "R":               round(R, 8),
        "delta":           delta,
        "final_alpha":     round(float(theta[0]), 6),
        "final_beta":      round(float(theta[1]), 6),
        "innovations_std": round(float(np.std(innovations)), 6),
        "n_bars":          n,
    }

    return (
        pd.Series(alpha_vals, index=close_a.index, name="alpha_kalman"),
        pd.Series(beta_vals,  index=close_a.index, name="beta_kalman"),
        diagnostics,
    )
```

---


## `src/analytics/characterize.py`

```python
"""
Spread Characterization

Half-life (OU AR(1) discretization), Hurst exponent (variance-ratio estimator),
and the window guardrail that converts half-life to a rolling window size.
"""
import numpy as np
import pandas as pd

# Guardrail constants matching configs/params_example.yaml
MIN_WINDOW     = 10
MAX_WINDOW     = 2000
DEFAULT_WINDOW = 60


def compute_half_life(spread: pd.Series) -> tuple[float, float]:
    """Estimate mean-reversion speed via OU AR(1) discretization.

    Regresses the first-difference of the spread on its lagged level:
        ΔS(t) = a + lambda * S(t-1) + epsilon

    The intercept (a) is included to avoid bias in spreads with non-zero mean.
    Without it, lambda — and therefore half-life — can be systematically wrong.

    Half-life = -ln(2) / lambda  when lambda < 0 (mean-reverting).

    Returns:
        (half_life, lambda_hat)
        half_life = np.inf  when lambda >= 0 (no mean reversion detected)
        half_life = np.nan  when the regression is numerically degenerate

    !! FLAG — interpretation caveats:
    1. Lambda is estimated on 1-min bars, so half-life is in MINUTES, not days.
       Divide by 390 if you want days.  The pipeline uses it directly as bars.
    2. For CMS/DUK formation (~47k bars) we expect a moderately short half-life.
       If HL > 240 (> 4 hours), the spread reverts very slowly and the engine
       will use the MAX_WINDOW = 240 cap.
    3. If spread variance is near-zero (constant series, trading halt), lstsq
       returns near-zero lambda -> very long HL -> capped at MAX_WINDOW (2000).
    """
    s = spread.dropna()
    lag   = s.shift(1)
    delta = s.diff()
    df_reg = pd.DataFrame({"delta": delta, "lag": lag}).dropna()

    if len(df_reg) < 2:
        return np.nan, np.nan

    X = np.column_stack([np.ones(len(df_reg)), df_reg["lag"].values])
    y = df_reg["delta"].values
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    _, lam = float(params[0]), float(params[1])

    if np.isnan(lam):
        return np.nan, np.nan
    if lam >= 0:
        # Non-negative lambda means no mean reversion (or explosive process).
        return np.inf, lam

    half_life = -np.log(2) / lam
    return half_life, lam


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Variance-ratio Hurst exponent estimator.

    For each lag τ in [2, max_lag), computes:
        tau(τ) = std(S(t+τ) − S(t))

    Then fits log(tau) ~ H * log(lag) via OLS. The slope H is the Hurst exponent.

    H < 0.5  → mean-reverting (favourable for pairs trading)
    H = 0.5  → random walk
    H > 0.5  → trending

    Returns np.nan if the estimator is degenerate (too few observations,
    zero-variance series, or a tau of 0 causing log(0)).

    !! FLAG — reliability caveats:
    1. This estimator is NOISY on short windows.  On the 47k-bar formation period
       it is reasonably stable, but on rolling sub-windows (Step 9) results
       should be treated as directional, not precise.
    2. The estimator is NOT bounded to [0, 1].  Extreme values (<0 or >1) signal
       a degenerate series or a very small sample — check the input.
    3. We use std (not var).  The slope from regressing log(std) on log(lag) is H
       directly (since std ~ lag^H → log(std) = H*log(lag) + const).
    4. max_lag=100 means we probe up to 100-minute (1.67h) autocorrelation structure.
       For pairs with longer-cycle mean reversion this may underestimate H.
    """
    vals = np.asarray(series.dropna(), dtype=np.float64)
    n = len(vals)

    if n < max_lag + 10:
        return np.nan

    lags = range(2, min(max_lag, n // 2))
    tau  = [np.std(vals[lag:] - vals[:-lag], ddof=1) for lag in lags]

    # Guard against zero-variance differences (constant spread segment)
    tau_arr = np.array(tau)
    if np.any(tau_arr <= 0):
        return np.nan

    slope = np.polyfit(np.log(list(lags)), np.log(tau_arr), 1)[0]
    return float(slope)


def window_from_half_life(
    half_life: float,
    min_w: int = MIN_WINDOW,
    max_w: int = MAX_WINDOW,
    default_w: int = DEFAULT_WINDOW,
) -> tuple[int, str]:
    """Convert a half-life (in bars) to a rolling window size with guardrails.

    Returns:
        (window, note)  where note explains any clamping or fallback applied.

    Guardrails (from research plan Step 3):
    - NaN, Inf, or <= 0  → use default_w  (no mean reversion detected)
    - Valid but < min_w  → clamp to min_w  (very fast reversion)
    - Valid but > max_w  → clamp to max_w  (very slow reversion)
    """
    if np.isnan(half_life) or np.isinf(half_life) or half_life <= 0:
        return default_w, f"half_life={half_life} invalid, using default {default_w}"

    raw = int(round(half_life))
    if raw < min_w:
        return min_w, f"half_life={half_life:.1f}, raw={raw} clamped up to min {min_w}"
    if raw > max_w:
        return max_w, f"half_life={half_life:.1f}, raw={raw} clamped down to max {max_w}"
    return raw, f"half_life={half_life:.1f}, window={raw}"
```

---


## `src/analytics/diagnostics.py`

```python
"""
Distribution Diagnostics

Empirical coverage table, tail-behavior statistics, and quantile
cross-validation between formation and trading periods.
"""
import pandas as pd
from scipy.stats import norm


# Thresholds used throughout the research plan
_COVERAGE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]


def empirical_coverage(zscore_series: pd.Series) -> pd.DataFrame:
    """Compare empirical Z-score coverage against normal-theory expectations.

    For each threshold t in [1.0, 1.5, 2.0, 2.5, 3.0], computes:
        - Normal theory: P(|Z| <= t) = 2*Phi(t) - 1
        - Empirical:     fraction of |z| <= t in the provided series
        - Gap:           theory - empirical (positive = fatter tails than normal)

    Args:
        zscore_series : Trading-period Z-scores (NaN will be dropped).

    Returns:
        DataFrame with columns [threshold, normal_pct, empirical_pct, gap_pct].

    !! FLAG — interpretation:
        A positive gap (normal > empirical) means more extreme values than a
        normal distribution predicts — fat tails.  For pairs-trading this is
        the key LTCM-relevant diagnostic: if ±2σ only covers 90% (vs 95.4%
        theory), the entry threshold fires more often than the model expects.
        Report the table as the centrepiece of the distribution analysis, not
        the normality p-values (which are near-guaranteed to reject on 48k bars).
    """
    z = zscore_series.dropna()
    rows = []
    for t in _COVERAGE_THRESHOLDS:
        theory_pct   = (2 * norm.cdf(t) - 1) * 100
        empirical_pct = (z.abs() <= t).mean() * 100
        rows.append({
            "threshold":     f"+-{t}",
            "normal_pct":    round(theory_pct,    2),
            "empirical_pct": round(empirical_pct, 2),
            "gap_pct":       round(theory_pct - empirical_pct, 2),
        })
    return pd.DataFrame(rows)


def tail_behavior(zscore_series: pd.Series) -> dict:
    """Compute tail-behaviour statistics for the Z-score distribution.

    Returns a dict with:
        n_obs        : Number of non-NaN observations
        mean         : Should be near 0 for a well-constructed spread
        std          : Should be near 1 (by construction, but rolling windows
                       introduce deviations)
        skewness     : Asymmetry; large magnitude signals directional drift
        excess_kurtosis : Kurtosis - 3; positive = fatter tails than normal
        pct_beyond_2 : Fraction of |z| > 2.0  (entry zone)
        pct_beyond_3 : Fraction of |z| > 3.0  (extreme events)
        q_025        : 2.5th percentile (lower tail)
        q_975        : 97.5th percentile (upper tail)
        abs_q95      : 95th percentile of |z|  (adaptive threshold basis)

    !! FLAG — excess_kurtosis on minute-bar data:
        Financial returns on 1-min data routinely show excess kurtosis > 5.
        High kurtosis does NOT mean the signal is broken — it reflects genuine
        fat-tail microstructure.  Report the number and connect it to the LTCM
        reading rather than treating it as a bug.
    """
    z = zscore_series.dropna()
    return {
        "n_obs":           int(len(z)),
        "mean":            round(float(z.mean()),     6),
        "std":             round(float(z.std(ddof=1)), 4),
        "skewness":        round(float(z.skew()),     4),
        "excess_kurtosis": round(float(z.kurtosis()), 4),   # pandas kurtosis is excess
        "pct_beyond_2":    round(float((z.abs() > 2.0).mean() * 100), 3),
        "pct_beyond_3":    round(float((z.abs() > 3.0).mean() * 100), 3),
        "q_025":           round(float(z.quantile(0.025)), 4),
        "q_975":           round(float(z.quantile(0.975)), 4),
        "abs_q95":         round(float(z.abs().quantile(0.95)), 4),
    }


def threshold_sensitivity_table(
    zscore: pd.Series,
    thresholds: list,
    cost_bps_rt: float = 60.0,
) -> pd.DataFrame:
    """Compare entry thresholds on a fixed Z-score series.

    For each entry_z in `thresholds`, runs the path-dependent state machine and
    reports trade count, average hold duration, and cost break-even.

    Args:
        zscore       : Z-score series (NaN rows hold position — same as state machine).
        thresholds   : List of entry_z values to compare, e.g. [1.5, 2.0, 2.5, 2.57].
        cost_bps_rt  : Cost per complete round-trip in basis points.
                       Default 60 bps = 4 one-way legs × 15 bps each.
                       A complete pairs trade involves 4 transactions:
                           open leg A, open leg B, close leg A, close leg B.

    Returns:
        DataFrame with columns:
            entry_z            : The threshold tested
            n_trades           : Number of round-trips entered
            avg_hold_bars      : Mean bars held per trade
            total_cost_bps     : n_trades × cost_bps_rt (total cost of running this threshold)
            breakeven_per_trade: Basis points of PnL needed per trade to cover costs (= cost_bps_rt)

    !! FLAG — interpretation:
        This table does NOT compute actual PnL (that is Week 3).  It shows the
        mechanical cost structure: how many trades, how long held, how much
        total cost.  The breakeven column is constant (= cost_bps_rt) because
        cost is per trade, not per bar.  The signal for choosing a threshold is:
        lower z fires more trades (higher total cost, lower average signal quality);
        higher z fires fewer trades (lower cost, only the clearest divergences).
    """
    from src.signals.state_machine import generate_positions, count_trades
    rows = []
    for z in thresholds:
        pos  = generate_positions(zscore, entry_z=float(z), exit_z=0.0)
        n    = count_trades(pos)
        hold = round((pos != 0).sum() / max(n, 1), 1)
        rows.append({
            "entry_z":             round(float(z), 4),
            "n_trades":            n,
            "avg_hold_bars":       hold,
            "total_cost_bps":      round(n * cost_bps_rt),
            "breakeven_per_trade": cost_bps_rt,
        })
    return pd.DataFrame(rows)


def quantile_crossval(
    z_formation: pd.Series,
    z_trading: pd.Series,
) -> dict:
    """Cross-validate the formation-derived threshold against the trading period.

    Computes the abs(Z) 95th percentile in both periods and flags any
    meaningful shift.  A large shift indicates the spread's volatility regime
    changed between formation and trading — the formation-derived adaptive
    threshold may be mis-calibrated.

    Returns:
        formation_q95  : abs(Z) 95th pct on formation period
        trading_q95    : abs(Z) 95th pct on trading period
        abs_drift      : |formation_q95 - trading_q95|
        regime_shift   : True if abs_drift > 0.5 (research plan threshold)

    !! FLAG — what a shift means:
        A formation_q95 of 2.56 and trading_q95 of 2.80 means the spread was
        more volatile in the trading period than expected from formation.
        The adaptive threshold of 2.56 (set from formation) will fire MORE
        trades in the trading period than intended.  Not a correctness error,
        but it must be stated in the Signal Logic Document.
    """
    q_f = float(z_formation.dropna().abs().quantile(0.95))
    q_t = float(z_trading.dropna().abs().quantile(0.95))
    drift = abs(q_f - q_t)
    return {
        "formation_q95": round(q_f, 4),
        "trading_q95":   round(q_t, 4),
        "abs_drift":     round(drift, 4),
        "regime_shift":  drift > 0.5,
    }
```

---


## `src/analytics/sensitivity.py`

```python
"""
Threshold Sensitivity

Sweeps entry thresholds and reports trade counts + daily rates.
Uses the state machine's count_trades() — NOT bar counts.
"""
import pandas as pd

from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions, count_trades


def run_sensitivity_analysis(
    spread_trading: pd.Series,
    window: int,
    thresholds: list[float],
    exit_z: float = 0.0,
) -> pd.DataFrame:
    """Sweep entry thresholds and record trade counts for each.

    All thresholds share the same Z-score series (computed once) and the
    same exit threshold.  Only the entry threshold varies.

    Args:
        spread_trading : Trading-period spread (formation hedge ratio applied).
        window         : Rolling window in bars — must be the same value used
                         for characterisation; do NOT re-derive here.
        thresholds     : List of entry_z values to test, e.g. [1.5, 2.0, 2.56, 2.5, 3.0].
                         The adaptive threshold should be included alongside fixed ones.
        exit_z         : Exit threshold (default 0.0 — zero-crossing exit).

    Returns:
        DataFrame with columns:
            threshold      : entry_z value tested
            label          : human-readable label (e.g. "fixed" / "adaptive")
            n_trades       : total round-trip entries over the trading period
            trading_days   : number of unique calendar dates in trading period
            trades_per_day : n_trades / trading_days

    !! FLAG — trade count vs bar count:
        n_trades counts state-machine entries (flat->non-flat transitions),
        not bars spent in position.  The research plan is explicit: use trade
        count for sensitivity analysis.  A threshold that gives 500 "trades"
        might mean 500 rapid scalps (GOOG/GOOGL-like) or 500 slow mean-
        reversion holds — the count alone doesn't distinguish them, but it
        is the correct metric for comparing threshold aggressiveness.

    !! FLAG — threshold ordering:
        Pass thresholds in ascending order so the output table reads naturally.
        If the adaptive threshold falls between two fixed values, insert it
        in the correct position in the list rather than appending it at the end.
    """
    zdf = compute_zscore(spread_trading, window=window)
    z   = zdf["zscore"]
    trading_days = spread_trading.index.normalize().nunique()

    rows = []
    for t in thresholds:
        pos     = generate_positions(z, entry_z=t, exit_z=exit_z)
        n       = count_trades(pos)
        rows.append({
            "threshold":      round(t, 4),
            "n_trades":       n,
            "trading_days":   trading_days,
            "trades_per_day": round(n / trading_days, 3) if trading_days else 0.0,
        })

    return pd.DataFrame(rows)
```

---


## `src/utils/config.py`

```python
"""
Configuration parsing using yaml
"""
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return the parsed dict.

    Raises FileNotFoundError if the path does not exist.
    Raises yaml.YAMLError if the file is malformed.
    No key validation is performed here — the caller owns that.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

---


## `src/utils/dates.py`

```python
"""
Date and Time manipulation helpers
"""
import pandas as pd


def split_periods(
    df: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    trading_start: str,
    trading_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an aligned pair DataFrame into formation and trading period slices.

    The no-lookahead invariant is enforced: formation_end must be strictly
    before trading_start. If violated, a ValueError is raised immediately
    before any slice is returned.

    Args:
        df              : Merged pair DataFrame with a UTC DatetimeIndex.
        formation_start : ISO date string, e.g. "2022-01-01"
        formation_end   : ISO date string, e.g. "2022-06-30"
        trading_start   : ISO date string, e.g. "2022-07-01"
        trading_end     : ISO date string, e.g. "2022-12-31"

    Returns:
        (formation, trading) — both are label-based slices of df retaining
        the same column structure and DatetimeIndex.
    """
    # Convert boundary strings to tz-aware Timestamps (accept pre-built Timestamps too)
    def _to_ts(val: str | pd.Timestamp) -> pd.Timestamp:
        if isinstance(val, pd.Timestamp):
            return val if val.tzinfo is not None else val.tz_localize("UTC")
        return pd.Timestamp(val, tz="UTC")

    fs = _to_ts(formation_start)
    fe = _to_ts(formation_end)
    ts = _to_ts(trading_start)
    te = _to_ts(trading_end)

    # Core no-lookahead guard
    if fe >= ts:
        raise ValueError(
            f"Lookahead violation: formation_end ({fe}) must be strictly before "
            f"trading_start ({ts}). Fix the date boundaries in your config."
        )

    formation = df.loc[fs:fe]
    trading = df.loc[ts:te]

    if formation.empty:
        raise ValueError(
            f"Formation slice is empty. Check formation dates [{fs}, {fe}] "
            f"against the DataFrame range [{df.index.min()}, {df.index.max()}]."
        )
    if trading.empty:
        raise ValueError(
            f"Trading slice is empty. Check trading dates [{ts}, {te}] "
            f"against the DataFrame range [{df.index.min()}, {df.index.max()}]."
        )

    return formation, trading
```

---


## `src/utils/io.py`

```python
"""
I/O helpers
"""
```

---


## `src/visuals/plots.py`

```python
"""
Plotting Utilities

Three diagnostic charts required for the Week 2 Signal Logic Document:
  1. plot_signal_validation  — 3-panel: prices / spread / Z-score with trade markers
  2. plot_rolling_hurst      — regime monitor over the trading period
  3. plot_qq_normal          — QQ-plot of Z-score vs standard normal

All functions save to a path and return the Figure for optional display.
Non-interactive backend (Agg) is set at import so plots work headlessly.
"""
import matplotlib
matplotlib.use("Agg")   # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import probplot


# ---------------------------------------------------------------------------
# 1. Three-panel signal validation chart
# ---------------------------------------------------------------------------

def plot_signal_validation(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    window: int,
    entry_z: float = 2.0,
    save_path: str = "outputs/figures/signal_validation.png",
    sample_week: str | None = None,
) -> plt.Figure:
    """Three-panel chart: prices, spread ±2σ band, Z-score with trade markers.

    Args:
        df          : Full pipeline output DataFrame containing columns:
                        close_a, close_b, spread, rolling_mean, rolling_std,
                        zscore, position.
        ticker_a/b  : Ticker labels for the legend.
        beta        : Hedge ratio (displayed in spread panel title).
        window      : Rolling window in bars (displayed in spread panel title).
        entry_z     : Entry threshold — drawn as horizontal lines on Z panel.
        save_path   : Output file path (PNG).
        sample_week : ISO date string for the start of the sample week to plot,
                      e.g. "2022-09-12".  If None, uses a week from ~2/3 through
                      the trading period (roughly November for Jul–Dec data).

    Panel layout
    ------------
    Panel 1 — Prices: close_a and close_b (normalised to 100 at window start
              so they fit on the same axis without the price-level difference
              dominating the visual).
    Panel 2 — Spread with rolling mean and ±entry_z*std band.
    Panel 3 — Z-score with threshold lines and entry/exit markers.
              Entry long  : green triangle up
              Entry short : red triangle down
              Exit        : black cross

    !! FLAG — sample_week selection:
        The chart covers a 5-day window.  Weeks with zero trades (all flat)
        are uninformative.  The auto-selection logic picks a week that contains
        at least one trade entry; if none is found in the default search window
        it falls back to the first week of the trading period.  Always visually
        inspect the saved chart — a bad week choice produces a flat Z-score
        panel with no markers, which looks like the engine is broken even when
        it is not.

    !! FLAG — normalised prices in Panel 1:
        Normalising to 100 removes the price-level difference between the two
        legs.  This is purely cosmetic for the validation chart.  It does NOT
        affect the spread or Z-score computation, which use raw log prices.
    """
    # -- Find a sample week that contains at least one trade ------------------
    if sample_week is None:
        sample_week = _find_active_week(df)

    week_end = pd.Timestamp(sample_week, tz=df.index.tz) + pd.Timedelta(days=7)
    week = df.loc[sample_week:week_end.strftime("%Y-%m-%d")]

    if week.empty or "position" not in week.columns:
        raise ValueError(
            f"sample_week '{sample_week}' produced an empty slice or df is missing "
            f"required columns. Columns present: {list(df.columns)}"
        )

    # -- Normalise prices to 100 at week start --------------------------------
    price_a = week["close_a"] / week["close_a"].iloc[0] * 100
    price_b = week["close_b"] / week["close_b"].iloc[0] * 100

    # -- Spread band ----------------------------------------------------------
    upper = week["rolling_mean"] + entry_z * week["rolling_std"]
    lower = week["rolling_mean"] - entry_z * week["rolling_std"]

    # -- Trade markers (entry/exit transitions) --------------------------------
    pos      = week["position"]
    prev_pos = pos.shift(1).fillna(0).astype("int8")
    entries_long  = week[(pos == 1)  & (prev_pos == 0)]
    entries_short = week[(pos == -1) & (prev_pos == 0)]
    exits         = week[(pos == 0)  & (prev_pos != 0)]

    # -- Build figure ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{ticker_a} / {ticker_b}  —  Signal Validation  "
        f"({sample_week})",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Prices
    ax0 = axes[0]
    ax0.plot(price_a.index, price_a.values, label=ticker_a, lw=1.2)
    ax0.plot(price_b.index, price_b.values, label=ticker_b, lw=1.2, alpha=0.85)
    ax0.set_ylabel("Price (normalised to 100)")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # Panel 2: Spread + band
    ax1 = axes[1]
    ax1.plot(week.index, week["spread"].values, label="Spread", lw=1.0, alpha=0.8)
    ax1.plot(week.index, week["rolling_mean"].values,
             color="black", lw=1.0, ls="--", label="Rolling mean")
    ax1.fill_between(week.index, lower.values, upper.values,
                     alpha=0.15, color="steelblue", label=f"+-{entry_z}*std band")
    ax1.set_ylabel("Log-price spread")
    ax1.set_title(
        f"Spread  (beta={beta:.4f}, window={window} bars)", fontsize=10
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 3: Z-score + markers
    ax2 = axes[2]
    ax2.plot(week.index, week["zscore"].values, lw=1.0, alpha=0.8, label="Z-score")
    ax2.axhline( entry_z, color="red",   ls="--", lw=1.0, label=f"+{entry_z}")
    ax2.axhline(-entry_z, color="green", ls="--", lw=1.0, label=f"-{entry_z}")
    ax2.axhline(0,        color="black", ls=":",  lw=0.8, alpha=0.5)

    if not entries_long.empty:
        ax2.scatter(entries_long.index, entries_long["zscore"].values,
                    marker="^", c="green", s=60, zorder=5, label="Entry long")
    if not entries_short.empty:
        ax2.scatter(entries_short.index, entries_short["zscore"].values,
                    marker="v", c="red", s=60, zorder=5, label="Entry short")
    if not exits.empty:
        ax2.scatter(exits.index, exits["zscore"].values,
                    marker="x", c="black", s=40, zorder=5, label="Exit")

    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left", fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # X-axis formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling Hurst regime monitor
# ---------------------------------------------------------------------------

def plot_rolling_hurst(
    hurst_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save_path: str = "outputs/figures/rolling_hurst.png",
) -> plt.Figure:
    """Rolling Hurst exponent over the trading period.

    Args:
        hurst_df : DataFrame with a DatetimeIndex and a 'hurst' column,
                   as returned by the rolling_hurst() helper in the pipeline.
        ticker_a/b : Ticker labels for the title.
        save_path  : Output file path.

    !! FLAG — estimator noise:
        On rolling sub-windows (typically 5000 bars = ~13 sessions) the
        Hurst estimator is noisy.  Values jumping between 0.3 and 0.7 within
        a few windows are normal and do not mean the spread is rapidly
        switching regimes.  Smooth interpretation: if the rolling H stays
        consistently above 0.5 for an extended stretch (say, a full month),
        that is a regime-shift signal worth noting.  Single-window spikes
        above 0.5 are noise.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(hurst_df.index, hurst_df["hurst"].values,
            lw=1.2, color="steelblue", label="Rolling H")
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="H = 0.5 (random walk)")
    ax.fill_between(hurst_df.index, hurst_df["hurst"].values, 0.5,
                    where=(hurst_df["hurst"].values < 0.5),
                    alpha=0.15, color="green", label="Mean-reverting (H < 0.5)")
    ax.fill_between(hurst_df.index, hurst_df["hurst"].values, 0.5,
                    where=(hurst_df["hurst"].values > 0.5),
                    alpha=0.15, color="red", label="Trending (H > 0.5)")

    pct_mr = (hurst_df["hurst"] < 0.5).mean() * 100
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  Rolling Hurst  "
        f"({pct_mr:.1f}% of windows H < 0.5)",
        fontsize=12,
    )
    ax.set_ylabel("Hurst exponent H")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. QQ-plot vs standard normal
# ---------------------------------------------------------------------------

def plot_qq_normal(
    zscore_series: pd.Series,
    ticker_a: str,
    ticker_b: str,
    save_path: str = "outputs/figures/qq_plot.png",
) -> plt.Figure:
    """QQ-plot of trading-period Z-scores against the standard normal.

    Deviations from the diagonal reveal fat tails (S-shape) or skewness
    (asymmetric deviation).  This is the visual companion to the empirical
    coverage table.

    !! FLAG — sample size:
        scipy.stats.probplot with 48k+ points is slow and produces an
        over-plotted chart.  A random sample of 5000 points is used for
        the visual without materially changing the diagnostic picture.
        The full series is still used for all numerical metrics.
    """
    z = zscore_series.dropna()

    # Subsample for visual clarity
    rng = np.random.default_rng(42)
    n_plot = min(5000, len(z))
    z_sample = pd.Series(
        rng.choice(z.values, size=n_plot, replace=False)
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    (osm, osr), (slope, intercept, _) = probplot(z_sample.values, dist="norm")
    ax.scatter(osm, osr, s=6, alpha=0.4, color="steelblue", label=f"Z-score (n={n_plot})")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="red", lw=1.5, ls="--", label="Normal reference line")

    ax.set_xlabel("Theoretical quantiles (standard normal)")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  QQ-Plot: Trading Period Z-Score vs Normal",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Kalman filter beta drift chart
# ---------------------------------------------------------------------------

def plot_kalman_beta(
    beta_series: pd.Series,
    ticker_a: str,
    ticker_b: str,
    formation_end: pd.Timestamp,
    static_beta: float,
    save_path: str = "outputs/figures/kalman_beta.png",
) -> plt.Figure:
    """Plot time-varying Kalman hedge ratio beta(t) over the full year.

    Args:
        beta_series   : Full-year beta series from kalman_hedge_ratio().
        ticker_a/b    : Ticker labels for title.
        formation_end : Last timestamp of the formation period — drawn as a
                        vertical boundary between formation and trading zones.
        static_beta   : Scalar OLS beta — drawn as a horizontal reference line.
        save_path     : Output file path.

    !! FLAG — interpretation:
        If the Kalman beta is flat and hugs the static OLS line, the relationship
        was stable in 2022 and the static approach was adequate.  If the beta
        drifts materially during the trading period, static OLS was introducing
        phantom signal — the Kalman spread will show better stationarity.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(beta_series.index, beta_series.values,
            lw=1.2, color="steelblue", label="Kalman beta(t)")
    ax.axhline(static_beta, color="red", ls="--", lw=1.2,
               label=f"Static OLS beta = {static_beta:.4f}")
    ax.axvline(formation_end, color="black", ls=":", lw=1.2, alpha=0.7,
               label="Formation | Trading boundary")

    # Shade the two periods
    x_min = beta_series.index[0]
    x_max = beta_series.index[-1]
    ax.axvspan(x_min, formation_end, alpha=0.05, color="blue",  label="Formation period")
    ax.axvspan(formation_end, x_max, alpha=0.05, color="green", label="Trading period")

    final_beta = float(beta_series.iloc[-1])
    drift      = round(final_beta - static_beta, 4)
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  Kalman Dynamic Beta"
        f"  (final={final_beta:.4f}, drift vs OLS={drift:+.4f})",
        fontsize=11,
    )
    ax.set_ylabel("Hedge ratio beta(t)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_active_week(df: pd.DataFrame) -> str:
    """Return the start date of a week in the middle third of df that has trades."""
    if "position" not in df.columns:
        # Fall back to first week
        return df.index[0].strftime("%Y-%m-%d")

    n = len(df)
    search_start = n // 3
    search_end   = (2 * n) // 3

    idx = df.index[search_start:search_end]
    pos = df["position"].iloc[search_start:search_end]
    prev = pos.shift(1).fillna(0)
    entries = idx[(pos != 0) & (prev == 0)]

    if len(entries) == 0:
        return df.index[0].strftime("%Y-%m-%d")

    # Pick the Monday (or first bar of that week) around the first entry found
    entry_ts = entries[0]
    week_start = entry_ts - pd.Timedelta(days=entry_ts.weekday())
    return week_start.strftime("%Y-%m-%d")


def _save(fig: plt.Figure, path: str) -> None:
    """Create parent directories if needed and save the figure."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


```

---


## `src/pipeline/run_week2.py`

```python
"""
Week 2 Pipeline Orchestration

Entry point that stitches together data loading, spread characterisation,
Z-score computation, and position generation for the primary and secondary
pairs defined in the config file.

Usage (from week2_signal_engine/):
    python src/pipeline/run_week2.py
    python src/pipeline/run_week2.py configs/params_example.yaml

Outputs:
    outputs/signals/signals_{A}_{B}.csv  — full trading-period signal DataFrame

!! FLAG — run_pipeline vs main():
    run_pipeline() is a pure computation function. It accepts pre-fitted
    parameters (alpha, beta, window) estimated on the formation period and
    applies them to the supplied price series.  It never touches formation
    data and never reads from disk.  Keep it that way — Week 3 PnL code
    will call run_pipeline() directly in tests.

!! FLAG — burn-in NaNs in output:
    The first (window // 2) rows of the returned DataFrame will have NaN
    z-scores.  Positions during this period are 0 (flat) because the state
    machine holds its current state on NaN — which starts at 0.  Do NOT
    drop these rows before saving: Week 3 needs the full aligned index.
    Use the `signal_valid` column to gate PnL computation rather than
    relying on implicit .dropna() calls that can silently mis-align indices.
"""
import os
import sys

# Allow running directly: python src/pipeline/run_week2.py
_PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_PIPELINE_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.signals.spread import compute_spread
from src.signals.zscore import compute_zscore, apply_session_warmup
from src.signals.state_machine import generate_positions, count_trades


# ---------------------------------------------------------------------------
# Core computation — pure function, no I/O
# ---------------------------------------------------------------------------

def run_pipeline(
    close_a,
    close_b,
    alpha: float | pd.Series,
    beta:  float | pd.Series,
    window: int,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    session_warmup: int = 0,
) -> pd.DataFrame:
    """Execute spread -> Z-score -> position pipeline on a price series.

    All parameters must be pre-estimated on the formation period. This
    function operates on the trading period only and never looks back at
    formation data.

    Args:
        close_a / close_b : pd.Series of closing prices (same DatetimeIndex).
        alpha             : Intercept — scalar (static OLS) or pd.Series (Kalman).
        beta              : Hedge ratio — scalar (static OLS) or pd.Series (Kalman).
                            Series must be indexed identically to close_a/close_b.
        window            : Rolling window in bars derived from formation half-life.
        entry_z           : Z-score threshold to enter a position (abs value).
        exit_z            : Z-score threshold to exit (0.0 = zero-crossing exit).
        session_warmup    : Bars to suppress Z-score after each session open.
                            0 = disabled.  30 = first 30 minutes flat each day.

    Returns:
        pd.DataFrame with columns:
            spread             : Log-price spread S(t) = log(A) - alpha - beta*log(B)
            rolling_mean       : Rolling mean of spread (window bars, ddof=1 std)
            rolling_std        : Rolling std of spread
            zscore             : (spread - rolling_mean) / rolling_std
            position           : int8 {-1, 0, +1} from the path-dependent state machine
            signal_valid       : bool — False during rolling burn-in (NaN z-score rows),
                                 True once the window has enough history.  Week 3 must
                                 gate all PnL computation on this column instead of using
                                 implicit .dropna() calls that can silently mis-align indices.
            position_executed  : int8 — position.shift(1), the position that was actually
                                 held at bar t (decided at bar t-1 close).  Week 3 PnL:
                                     return[t] = position_executed[t] * pct_change[t]
                                 Using position[t] instead would introduce 1-bar lookahead
                                 on every trade on 1-minute data.
    """
    spread = compute_spread(close_a, close_b, alpha, beta)
    zdf    = compute_zscore(spread, window=window)
    if session_warmup > 0:
        zdf = apply_session_warmup(zdf, session_warmup)
    pos    = generate_positions(zdf["zscore"], entry_z=entry_z, exit_z=exit_z)

    # signal_valid: True once the rolling window has enough history
    signal_valid = ~zdf["zscore"].isna()

    # position_executed: shift by 1 bar — the position held *during* bar t
    # was decided at bar t-1.  fillna(0) sets bar 0 to flat (no prior bar).
    position_executed = pos.shift(1).fillna(0).astype("int8")

    out = zdf.copy()
    out.insert(0, "spread", spread)
    out["position"]          = pos
    out["signal_valid"]      = signal_valid
    out["position_executed"] = position_executed
    return out[["spread", "rolling_mean", "rolling_std", "zscore",
                "position", "signal_valid", "position_executed"]]


# ---------------------------------------------------------------------------
# Orchestration — handles I/O, config, and pair loop
# ---------------------------------------------------------------------------

def _default_config_path():
    return os.path.join(_PROJECT_ROOT, "configs", "params_example.yaml")


def main(config_path: str | None = None) -> None:
    """Load config, run the full pipeline for primary and secondary pairs,
    save signal CSVs, and print a summary.

    Args:
        config_path : Path to YAML config.  Defaults to configs/params_example.yaml
                      (resolved relative to the project root).
    """
    from src.utils.config import load_config
    from src.data.loaders import load_pair
    from src.utils.dates import split_periods
    from src.signals.spread import estimate_hedge_ratio
    from src.analytics.characterize import compute_half_life, window_from_half_life
    from src.analytics.diagnostics import threshold_sensitivity_table

    cfg_path = config_path or _default_config_path()
    cfg      = load_config(cfg_path)
    dates    = cfg["dates"]
    engine   = cfg.get("engine", {})
    entry_z      = engine.get("entry_z_fixed", 2.0)
    exit_z       = engine.get("exit_z_fixed",  0.0)
    use_kalman   = bool(engine.get("use_kalman", False))
    kalman_delta = float(engine.get("kalman_delta", 1e-5))
    session_warmup = int(engine.get("session_open_warmup", 0))

    data_dir    = os.path.join(_PROJECT_ROOT, "data", "raw")
    signals_dir = os.path.join(_PROJECT_ROOT, "outputs", "signals")
    os.makedirs(signals_dir, exist_ok=True)

    pairs_to_run = [
        ("primary",   cfg["pairs"]["primary"]),
        ("secondary", cfg["pairs"]["secondary"]),
    ]

    mode_label = "KALMAN" if use_kalman else "STATIC OLS"
    print("\n" + "=" * 60)
    print(f"WEEK 2 SIGNAL ENGINE  [{mode_label}]")
    print("=" * 60)

    for category, (ta, tb) in pairs_to_run:
        print(f"\n--- {category.upper()}: {ta}/{tb} ---")

        merged, audit = load_pair(data_dir, ta, tb)
        formation, trading = split_periods(merged, **dates)

        # Static OLS — always estimated (used as reference even in Kalman mode)
        alpha_ols, beta_ols = estimate_hedge_ratio(
            formation["close_a"], formation["close_b"]
        )

        if use_kalman:
            # Run filter on FULL year — formation warms it up, trading continues
            from src.signals.kalman import kalman_hedge_ratio
            alpha_full, beta_full, kf_diag = kalman_hedge_ratio(
                merged["close_a"], merged["close_b"], delta=kalman_delta
            )
            # Slice trading period — no re-initialization at the boundary
            alpha_trade  = alpha_full.loc[trading.index]
            beta_trade   = beta_full.loc[trading.index]
            # Window estimation: use OLS formation spread, NOT the Kalman spread.
            # The Kalman posterior spread at bar t uses theta_t (state after
            # observing bar t) — this makes it approximately the scaled Kalman
            # innovation, which is white noise by construction (HL≈0.6 bars).
            # Feeding that into window_from_half_life() produces window=10 (min),
            # which generates ~1,300 trades/period at 10.7/day — not pairs trading.
            # Option A: anchor the Kalman window to the OLS formation half-life so
            # both methods operate at the same mean-reversion timescale.
            spread_f = compute_spread(
                formation["close_a"], formation["close_b"], alpha_ols, beta_ols
            )
            csv_suffix = "_kalman"
            print(f"  [KALMAN]  delta={kalman_delta}  "
                  f"final_beta={kf_diag['final_beta']:.6f}  "
                  f"(OLS beta={beta_ols:.6f}, "
                  f"drift={kf_diag['final_beta'] - beta_ols:+.6f})")
        else:
            alpha_trade = alpha_ols
            beta_trade  = beta_ols
            spread_f    = compute_spread(
                formation["close_a"], formation["close_b"], alpha_ols, beta_ols
            )
            csv_suffix = ""

        hl, _        = compute_half_life(spread_f)
        window, note = window_from_half_life(hl)

        # Formation adaptive threshold (95th pct of abs Z)
        zdf_f      = compute_zscore(spread_f, window=window)
        adaptive_z = round(float(zdf_f["zscore"].dropna().abs().quantile(0.95)), 4)

        # Trading-period signals (uses formation-derived parameters, no leakage)
        result_df = run_pipeline(
            trading["close_a"], trading["close_b"],
            alpha=alpha_trade, beta=beta_trade, window=window,
            entry_z=entry_z, exit_z=exit_z,
            session_warmup=session_warmup,
        )

        # Save signal CSV
        out_path = os.path.join(signals_dir, f"signals_{ta}_{tb}{csv_suffix}.csv")
        result_df.to_csv(out_path)

        # Summary
        n_trades     = count_trades(result_df["position"])
        trading_days = result_df.index.normalize().nunique()
        tpd          = n_trades / trading_days if trading_days else 0.0
        z_std        = round(float(result_df["zscore"].dropna().std(ddof=1)), 4)
        # Window-truncation correction: rolling std underestimates true variance
        # when window << half-life.  The true 2-sigma event in this Z-score
        # distribution sits at z = 2.0 * z_std.  Compare to adaptive_z.
        true_2sigma  = round(2.0 * z_std, 4)

        print(f"  alpha (OLS) = {alpha_ols:.6f}")
        print(f"  beta  (OLS) = {beta_ols:.6f}")
        print(f"  half-life   = {hl:.1f} bars  ({note})")
        print(f"  window      = {window} bars")
        print(f"  adaptive_z  = {adaptive_z}  (fixed entry_z = {entry_z})")
        print(f"  z_std       = {z_std}  (ideal=1.0; window-truncation inflates to {z_std})")
        print(f"  true_2sigma = {true_2sigma}  (2.0 * z_std — actual 2-sigma bar in this distribution)")
        print(f"  trades      = {n_trades}  over {trading_days} days  ({tpd:.2f}/day)")
        print(f"  signal CSV  -> {out_path}")

        # Threshold sensitivity table (formation Z-score, cost = 60 bps/round-trip)
        tst = threshold_sensitivity_table(
            zdf_f["zscore"].dropna(),
            thresholds=[1.5, 2.0, 2.5, adaptive_z],
        )
        print(f"\n  Threshold sensitivity (formation period, 60 bps/round-trip cost):")
        print(f"  {'entry_z':>8}  {'n_trades':>9}  {'avg_hold':>9}  {'total_cost_bps':>14}  {'breakeven_bps':>13}")
        for _, row in tst.iterrows():
            marker = " <-- fixed" if row["entry_z"] == entry_z else (
                     " <-- adaptive" if row["entry_z"] == adaptive_z else "")
            print(f"  {row['entry_z']:>8.4f}  {row['n_trades']:>9}  "
                  f"{row['avg_hold_bars']:>9.1f}  {row['total_cost_bps']:>14.0f}  "
                  f"{row['breakeven_per_trade']:>13.1f}{marker}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print()


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path=cfg_arg)
```

---


## `scripts/run_all_pairs_diagnostics.py`

```python
"""
Multi-pair Chunk 4 Diagnostics
Runs the full diagnostic pipeline for all 11 pairs from params_example.yaml,
saves 3 plots per pair, and prints a comparative summary table.

Run from: week2_signal_engine/
  python run_all_pairs_diagnostics.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import compute_half_life, hurst_exponent, window_from_half_life
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions, count_trades
from src.analytics.diagnostics import empirical_coverage, tail_behavior, quantile_crossval
from src.visuals.plots import plot_signal_validation, plot_rolling_hurst, plot_qq_normal

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "figures", "all_pairs")

# Week 1 Engle-Granger scan results — pairs were selected from this universe scan.
# Provides the formal cointegration test statistics that the Week 2 engine relies on.
WEEK1_SCAN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Week 1",
    "outputs", "pair_scan_results", "full_pair_scan_results.parquet"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_pairs(cfg):
    """Flatten config pairs into list of (category, ta, tb)."""
    pairs = []
    p = cfg["pairs"]
    pairs.append(("primary",      p["primary"][0],   p["primary"][1]))
    pairs.append(("secondary",    p["secondary"][0],  p["secondary"][1]))
    for alt in p.get("alternatives", []):
        pairs.append(("alternative", alt[0], alt[1]))
    for bm in p.get("benchmarks", []):
        pairs.append(("benchmark",   bm[0],  bm[1]))
    for nc in p.get("negative_controls", []):
        pairs.append(("neg_control", nc[0],  nc[1]))
    return pairs


def rolling_hurst_series(spread, rolling_window=5000, step=500, max_lag=100):
    vals = spread.dropna()
    rows, timestamps = [], []
    for start in range(0, len(vals) - rolling_window, step):
        end = start + rolling_window
        H   = hurst_exponent(vals.iloc[start:end], max_lag=max_lag)
        rows.append({"hurst": H})
        timestamps.append(vals.index[min(end - 1, len(vals) - 1)])
    if not rows:
        return pd.DataFrame(columns=["hurst"])
    df_h = pd.DataFrame(rows)
    df_h.index = pd.DatetimeIndex(timestamps[:len(df_h)])
    return df_h


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair(ta, tb, dates, label):
    result = {"pair": f"{ta}/{tb}", "label": label, "status": "OK", "error": None}
    try:
        merged, _ = load_pair(DATA_DIR, ta, tb)
        formation, trading = split_periods(merged, **dates)

        fa, fb       = formation["close_a"], formation["close_b"]
        alpha, beta  = estimate_hedge_ratio(fa, fb)
        spread_f     = compute_spread(fa, fb, alpha, beta)
        hl, _        = compute_half_life(spread_f)
        H_form       = hurst_exponent(spread_f)
        window, _    = window_from_half_life(hl)

        zdf_f    = compute_zscore(spread_f, window=window)
        z_f      = zdf_f["zscore"].dropna()
        adaptive = round(float(z_f.abs().quantile(0.95)), 4)

        spread_t = compute_spread(trading["close_a"], trading["close_b"], alpha, beta)
        zdf_t    = compute_zscore(spread_t, window=window)
        z_t      = zdf_t["zscore"]

        pos_fixed = generate_positions(z_t, entry_z=2.0,      exit_z=0.0)
        pos_adapt = generate_positions(z_t, entry_z=adaptive,  exit_z=0.0)
        n_fixed   = count_trades(pos_fixed)
        n_adapt   = count_trades(pos_adapt)

        cov_df  = empirical_coverage(z_t)
        cov_2s  = float(cov_df.loc[cov_df["threshold"] == "+-2.0", "empirical_pct"].values[0])
        gap_2s  = float(cov_df.loc[cov_df["threshold"] == "+-2.0", "gap_pct"].values[0])
        tb_stat = tail_behavior(z_t)
        xval    = quantile_crossval(z_f, z_t)

        hurst_df = rolling_hurst_series(spread_t)
        pct_mr   = (hurst_df["hurst"] < 0.5).mean() * 100 if len(hurst_df) > 0 else float("nan")

        # --- Plots ---
        pair_fig_dir = os.path.join(FIG_DIR, f"{ta}_{tb}")
        os.makedirs(pair_fig_dir, exist_ok=True)

        df_pipe = trading[["close_a", "close_b"]].copy()
        df_pipe["spread"]       = spread_t
        df_pipe["rolling_mean"] = zdf_t["rolling_mean"]
        df_pipe["rolling_std"]  = zdf_t["rolling_std"]
        df_pipe["zscore"]       = z_t
        df_pipe["position"]     = pos_fixed

        plot_errors = []
        for plot_fn, kwargs, fname in [
            (plot_signal_validation,
             dict(df=df_pipe, ticker_a=ta, ticker_b=tb, beta=beta, window=window, entry_z=2.0),
             "signal_validation.png"),
            (plot_rolling_hurst,
             dict(hurst_df=hurst_df, ticker_a=ta, ticker_b=tb) if len(hurst_df) > 0 else None,
             "rolling_hurst.png"),
            (plot_qq_normal,
             dict(zscore_series=z_t, ticker_a=ta, ticker_b=tb),
             "qq_plot.png"),
        ]:
            if kwargs is None:
                continue
            try:
                kwargs["save_path"] = os.path.join(pair_fig_dir, fname)
                plot_fn(**kwargs)
            except Exception as e:
                plot_errors.append(f"{fname}: {e}")

        hl_display = round(hl, 1) if (not pd.isna(hl) and hl != float("inf")) else str(hl)
        H_display  = round(H_form, 3) if not pd.isna(H_form) else float("nan")

        result.update({
            "beta":           round(beta, 4),
            "half_life":      hl_display,
            "H_form":         H_display,
            "window":         window,
            "adaptive_z":     adaptive,
            "std":            tb_stat["std"],
            "excess_kurt":    tb_stat["excess_kurtosis"],
            "skewness":       tb_stat["skewness"],
            "cov_2s_pct":     round(cov_2s, 2),
            "gap_2s_pct":     round(gap_2s, 2),
            "regime_shift":   xval["regime_shift"],
            "drift":          xval["abs_drift"],
            "n_fixed":        n_fixed,
            "n_adapt":        n_adapt,
            "pct_mr":         round(pct_mr, 1),
            "n_hurst_wins":   len(hurst_df),
            "plot_errors":    plot_errors,
        })
        if plot_errors:
            result["status"] = "WARN"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"]  = str(e)
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg   = load_config(CONFIG_PATH)
    dates = cfg["dates"]
    pairs = get_all_pairs(cfg)

    # Load Week 1 Engle-Granger scan results (pairs were selected from this scan)
    week1_scan = None
    if os.path.exists(WEEK1_SCAN_PATH):
        week1_scan = pd.read_parquet(WEEK1_SCAN_PATH)[
            ["ticker_a", "ticker_b", "coint_tstat", "raw_pval"]
        ]
    else:
        print(f"  [WARN] Week 1 scan not found at: {WEEK1_SCAN_PATH}")

    print("\n" + "=" * 70)
    print(f"MULTI-PAIR DIAGNOSTICS  ({len(pairs)} pairs)")
    print("=" * 70)

    results = []
    for label, ta, tb in pairs:
        print(f"  [{label.upper():12s}]  {ta}/{tb} ...", end="", flush=True)
        r = run_pair(ta, tb, dates, label)

        # Attach Week 1 EG cointegration evidence
        if week1_scan is not None:
            row = week1_scan[
                (week1_scan["ticker_a"] == ta) & (week1_scan["ticker_b"] == tb)
            ]
            if not row.empty:
                r["eg_tstat"] = round(float(row["coint_tstat"].iloc[0]), 4)
                r["eg_pval"]  = round(float(row["raw_pval"].iloc[0]),    6)
            else:
                r["eg_tstat"] = float("nan")
                r["eg_pval"]  = float("nan")

        results.append(r)
        if r["status"] == "OK":
            print(" OK")
        elif r["status"] == "WARN":
            print(f" WARN  (plot issues: {len(r['plot_errors'])})")
        else:
            print(f" ERROR: {r['error']}")

    # ── Comparative table ──────────────────────────────────────────────────
    print("\n\n" + "=" * 130)
    print("COMPARATIVE SUMMARY TABLE")
    print("=" * 130)

    cols = [
        ("Pair",       "pair",         12),
        ("Category",   "label",        12),
        ("EG_tstat",   "eg_tstat",      9),
        ("EG_pval",    "eg_pval",       9),
        ("Beta",       "beta",          7),
        ("HL(bars)",   "half_life",     9),
        ("H_form",     "H_form",        7),
        ("Window",     "window",        7),
        ("Adap_Z",     "adaptive_z",    7),
        ("Std",        "std",           6),
        ("Kurt",       "excess_kurt",   7),
        ("Skew",       "skewness",      7),
        ("Cov2s%",     "cov_2s_pct",    8),
        ("Gap2s%",     "gap_2s_pct",    8),
        ("Drift",      "drift",         7),
        ("Regime?",    "regime_shift",  8),
        ("N_fix",      "n_fixed",       7),
        ("N_adp",      "n_adapt",       7),
        ("%MR_win",    "pct_mr",        8),
    ]

    header = "  ".join(f"{title:<{w}}" for title, _, w in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        if r["status"] == "ERROR":
            print(f"  {r['pair']:<12}  {r['label']:<12}  ERROR: {r['error']}")
            continue
        row = "  ".join(f"{str(r.get(k, 'N/A')):<{w}}" for _, k, w in cols)
        print(row)

    # ── Interpretation notes ───────────────────────────────────────────────
    ok = [r for r in results if r["status"] != "ERROR"]
    mr_pairs     = [r for r in ok if isinstance(r.get("H_form"), float) and r["H_form"] < 0.5]
    regime_pairs = [r for r in ok if r.get("regime_shift")]
    kurt_pairs   = [r for r in ok if r.get("excess_kurt", 0) > 3]
    fat_tail_pairs = [r for r in ok if r.get("gap_2s_pct", 0) > 5]
    high_trade   = [r for r in ok if r.get("n_fixed", 0) > 300]

    print("\n\n" + "=" * 70)
    print("INTERPRETATION NOTES")
    print("=" * 70)
    print(f"  Mean-reverting formation spread (H < 0.5)   : {len(mr_pairs)}/{len(ok)} pairs")
    for r in mr_pairs:
        print(f"    {r['pair']:<14}  H={r['H_form']}")

    print(f"\n  Regime shift in volatility (drift > 0.5)    : {len(regime_pairs)}/{len(ok)} pairs")
    for r in regime_pairs:
        print(f"    {r['pair']:<14}  drift={r['drift']}")

    print(f"\n  Genuine fat tails (excess kurtosis > 3)     : {len(kurt_pairs)}/{len(ok)} pairs")
    for r in kurt_pairs:
        print(f"    {r['pair']:<14}  kurt={r['excess_kurt']}")

    print(f"\n  Significant coverage gap at +-2s (gap > 5%) : {len(fat_tail_pairs)}/{len(ok)} pairs")
    for r in fat_tail_pairs:
        print(f"    {r['pair']:<14}  cov={r['cov_2s_pct']:.1f}%  gap={r['gap_2s_pct']:+.2f}%")

    print(f"\n  High-frequency pairs (> 300 trades, fixed)  : {len(high_trade)}/{len(ok)} pairs")
    for r in high_trade:
        print(f"    {r['pair']:<14}  n_fixed={r['n_fixed']}")

    errors = [r for r in results if r["status"] == "ERROR"]
    if errors:
        print(f"\n  FAILED ({len(errors)}):")
        for r in errors:
            print(f"    {r['pair']}: {r['error']}")
    else:
        print("\n  All pairs completed without errors.")

    print(f"\n  Figures saved under: {FIG_DIR}")
    print()


if __name__ == "__main__":
    main()
```

---


## `tests/test_spread.py`

```python
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
```

---


## `tests/test_zscore.py`

```python
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
```

---


## `tests/test_state_machine.py`

```python
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
```

---


## `tests/test_kalman.py`

```python
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
```

---

