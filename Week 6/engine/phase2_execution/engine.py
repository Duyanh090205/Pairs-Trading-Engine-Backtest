"""
Phase 2 — Execution Engine (V3.0)
==================================

V3.0 changes vs V2.0:
  - Static spread (R4/F4): replaces misnamed Kalman with explicit static spread
    (hedge ratio fixed at formation; no posterior). rebalance_log stays empty.
  - Factor-residual input (F1): log_close fed in is residual log-price, not raw.
  - Time-of-day mask (A1): blocks ENTRIES during 11:00-11:59 + 15:30-15:59 ET.
  - Hard stop-loss (A2): signed exit at |Z|>=5.5 in catastrophic-widen direction.
  - Z-velocity gate (A3): blocks entry when |dZ/dt| over 10-bar lookback >= 0.05.
  - Vol-target sizing (A4): notional per pair = target_daily_vol / spread_vol_per_$.
  - Dynamic Z entry (F2): _state_machine accepts per-bar z_entry_array, allowing
    fixed / vol-conditional / HL-conditional Z entry strategies.

Per fold, per pair (V3.0):
  1. apply_ticker_concentration_cap   — max 5 pairs/ticker
  2. compute_static_spread on 1-min   → spread series
  3. Z-score (formation-locked or rolling, with session warmup)
  4. build_block_entry_mask           — time-of-day mask
  5. build z_entry_array              — per-bar entry threshold (fixed/vol/hl)
  6. _state_machine                   — entries, exits, hard SL, Z-velocity gate
  7. 1-bar execution lag
  8. compute_vol_target_notional      — vol-target sizing
  9. Threshold rebalance (DEAD branch under static spread)

Returns bar-level trade log per pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

from engine.phase2_execution.static_spread import (
    compute_static_spread,
    compute_formation_z_params,
)
from engine.utils.stats import rolling_zscore
from engine.z_strategies import build_z_entry_array

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BARS_PER_DAY_1MIN = 390
_SESSION_WARMUP_BARS = 30       # first 30 bars each session → NaN zscore
_Z_WINDOW_CAP = 2000            # cap on rolling Z window in bars
_REBALANCE_THRESHOLD = 0.10     # 10% beta drift triggers rebalance (DEAD under V3 static)
_REBALANCE_DEAD_BAND = 0.05     # 5% = X/2, hysteresis dead band
_REBALANCE_COST_BPS = 0.0030    # 30 bps one-side on delta shares
_N_OPEN_PAIRS_MAX = 50          # default portfolio cap
_TOTAL_CAPITAL = 1_000_000.0    # $1M notional (normalised)
_MAX_PAIRS_PER_TICKER = 5       # concentration cap

# V3.0 ADD: thresholds (fixed at design time — do not sweep)
_HARD_SL_Z = 5.5                # A2 — signed catastrophic widen exit
_Z_VELOCITY_LOOKBACK = 10       # A3 — bars (10 × 1-min = 10 min lookback)
_Z_VELOCITY_THRESH = 0.05       # A3 — Z-units per bar; |dZ/dt| above this blocks entry
_VOL_TARGET_DAILY = 200.0       # A4 — $ daily P&L vol target per pair
_VOL_TARGET_FLOOR = 10_000.0    # A4 — min notional per pair
_VOL_TARGET_CAP = 40_000.0      # A4 — max notional per pair

# V3.0 ADD: kill-zone definition (A1) — 11:00–11:59 and 15:30–15:59 ET inclusive
_KILL_ZONE_RANGES = (
    ((11,  0), (11, 59)),
    ((15, 30), (15, 59)),
)


# ---------------------------------------------------------------------------
# Portfolio construction: per-ticker concentration cap
# ---------------------------------------------------------------------------

def apply_ticker_concentration_cap(
    pairs_df: pd.DataFrame,
    max_pairs_per_ticker: int = _MAX_PAIRS_PER_TICKER,
) -> pd.DataFrame:
    """
    Keep at most max_pairs_per_ticker pairs per ticker (ranked by johansen_pval asc).
    Addresses folds with 5,000+ pairs where a single ticker appears in 100+ pairs.
    """
    if pairs_df.empty:
        return pairs_df

    ranked = pairs_df.sort_values("johansen_pval").reset_index(drop=True)
    ticker_count: dict[str, int] = {}
    keep_idx: list[int] = []

    for idx, row in ranked.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        ca = ticker_count.get(ta, 0)
        cb = ticker_count.get(tb, 0)
        if ca < max_pairs_per_ticker and cb < max_pairs_per_ticker:
            keep_idx.append(idx)
            ticker_count[ta] = ca + 1
            ticker_count[tb] = cb + 1

    result = ranked.loc[keep_idx].reset_index(drop=True)
    if len(result) < len(ranked):
        log.info(
            "Concentration cap: %d -> %d pairs (max %d per ticker)",
            len(ranked), len(result), max_pairs_per_ticker,
        )
    return result


# ---------------------------------------------------------------------------
# Session warmup mask
# ---------------------------------------------------------------------------

def _session_warmup_mask(index: pd.DatetimeIndex, warmup_bars: int) -> np.ndarray:
    """True where bar is within first warmup_bars of its session. Vectorized."""
    n = len(index)
    dates = index.normalize().view(np.int64)  # compare as ints, no Python date objects
    mask = np.zeros(n, dtype=bool)
    # Session starts: index 0 and wherever date changes
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = dates[1:] != dates[:-1]
    starts = np.where(is_new)[0]
    for s in starts:
        mask[s: s + warmup_bars] = True
    return mask


# ---------------------------------------------------------------------------
# V3.0 A1 — Time-of-day mask (block entries during kill zones)
# ---------------------------------------------------------------------------

def build_block_entry_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    True where new entries are blocked (kill zones from Week 5 §8):
        11:00-11:59 ET and 15:30-15:59 ET inclusive.
    Exits in these windows are still allowed — only entries are gated.
    """
    # ---- HARD STOP ----
    if index.tz is None:
        raise ValueError("build_block_entry_mask: index must be tz-aware (US/Eastern)")
    tz_name = str(index.tz)
    if "US/Eastern" not in tz_name and "America/New_York" not in tz_name:
        raise ValueError(
            f"build_block_entry_mask: index.tz='{tz_name}', expected US/Eastern"
        )

    n = len(index)
    hours = index.hour.to_numpy()
    minutes = index.minute.to_numpy()
    mins_since_midnight = hours * 60 + minutes

    mask = np.zeros(n, dtype=bool)
    for (h_lo, m_lo), (h_hi, m_hi) in _KILL_ZONE_RANGES:
        lo = h_lo * 60 + m_lo
        hi = h_hi * 60 + m_hi
        mask |= (mins_since_midnight >= lo) & (mins_since_midnight <= hi)

    # ---- INVARIANT ----
    assert mask.dtype == bool
    assert mask.shape == (n,)
    return mask


# ---------------------------------------------------------------------------
# V3.0 A4 — Vol-target notional sizing
# ---------------------------------------------------------------------------

def compute_vol_target_notional(
    spread: np.ndarray,
    target_daily_vol: float = _VOL_TARGET_DAILY,
    floor: float = _VOL_TARGET_FLOOR,
    cap: float = _VOL_TARGET_CAP,
) -> float:
    """
    Compute per-pair notional so that expected daily $ P&L vol = target_daily_vol.

    notional = target_daily_vol / spread_daily_vol_per_$ (clipped to [floor, cap]).
    """
    # ---- HARD STOPS ----
    if not isinstance(spread, np.ndarray):
        raise ValueError("compute_vol_target_notional: spread must be ndarray")
    if target_daily_vol <= 0:
        raise ValueError(
            f"compute_vol_target_notional: target_daily_vol must be > 0, got {target_daily_vol}"
        )
    if not (0 < floor < cap):
        raise ValueError(
            f"compute_vol_target_notional: need 0 < floor < cap, got floor={floor}, cap={cap}"
        )
    if len(spread) < _BARS_PER_DAY_1MIN:
        raise ValueError(
            f"compute_vol_target_notional: need >={_BARS_PER_DAY_1MIN} bars, got {len(spread)}"
        )

    spread_finite = spread[np.isfinite(spread)]
    if len(spread_finite) < _BARS_PER_DAY_1MIN:
        return floor

    # Per-bar spread return std → annualize to daily by sqrt(bars/day)
    bar_returns = np.diff(spread_finite)
    if len(bar_returns) < 2:
        return floor
    spread_bar_std = float(np.std(bar_returns))
    if spread_bar_std < 1e-12:
        return cap   # essentially flat spread => use cap (degenerate; A2 SL will kill if needed)

    spread_daily_vol_per_dollar = spread_bar_std * np.sqrt(_BARS_PER_DAY_1MIN)
    notional = target_daily_vol / spread_daily_vol_per_dollar

    # ---- INVARIANTS / CLIP ----
    notional = float(max(floor, min(cap, notional)))
    assert np.isfinite(notional)
    assert floor <= notional <= cap
    return notional


# ---------------------------------------------------------------------------
# V3.0 — Numba state machine with dynamic Z, time-mask, hard SL, Z-velocity gate
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _state_machine(
    zscores: np.ndarray,
    z_entry_array: np.ndarray,
    block_entry: np.ndarray,
) -> np.ndarray:
    """
    V3.0 signal state machine.

    Args:
        zscores:        per-bar Z-score (np.nan during warmup or missing data)
        z_entry_array:  per-bar entry threshold (from Z strategy: fixed/vol/hl)
        block_entry:    per-bar boolean — True means kill-zone, no new entry

    State:
        +1 = long A short B, -1 = short A long B, 0 = flat

    Transitions:
        entry  : state==0 + |z|>z_entry[i] + not block_entry[i] + Z-velocity OK
        exit   : zero-crossing (signed)
        hard SL: signed widen beyond _HARD_SL_Z (A2)
        nan Z  : hold current state, no transitions
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    state = np.int8(0)

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            positions[i] = state
            continue

        entry_z = z_entry_array[i]

        # ---- A2: Hard SL (catastrophic widen) ----
        if state == 1 and z <= -_HARD_SL_Z:
            state = np.int8(0)
            positions[i] = state
            continue
        if state == -1 and z >= _HARD_SL_Z:
            state = np.int8(0)
            positions[i] = state
            continue

        # ---- Exit on zero-crossing ----
        if state != 0:
            if (state == 1 and z >= 0.0) or (state == -1 and z <= 0.0):
                state = np.int8(0)
            positions[i] = state
            continue

        # ---- Entry path (state == 0) ----
        if block_entry[i]:
            # A1 kill-zone: no entry
            positions[i] = state
            continue

        # A3: Z-velocity gate — block entry when spread is moving rapidly
        # (likely factor momentum carrying through threshold, not a settled dislocation).
        # Restored after switching to rolling-Z (was over-aggressive with formation-locked Z).
        if i >= _Z_VELOCITY_LOOKBACK:
            z_prev = zscores[i - _Z_VELOCITY_LOOKBACK]
            if not np.isnan(z_prev):
                z_velocity = (z - z_prev) / _Z_VELOCITY_LOOKBACK
                if z_velocity >= _Z_VELOCITY_THRESH or z_velocity <= -_Z_VELOCITY_THRESH:
                    positions[i] = state
                    continue

        if z < -entry_z:
            state = np.int8(1)
        elif z > entry_z:
            state = np.int8(-1)

        positions[i] = state

    return positions




# ---------------------------------------------------------------------------
# EOS (end-of-session) flatten
# ---------------------------------------------------------------------------

def _apply_eos_flatten(
    positions: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Force signal to 0 from 15:55 ET through end-of-session each day.
    Zeros all bars from the 15:55 bar (inclusive) to the session close.

    With 1-bar execution lag, zeroing only 15:55 leaves signal non-zero at
    15:56-15:59, which propagates into position at 15:57-16:00. Zeroing the
    full tail ensures position is 0 from 15:56 onward for every session.

    Falls back to zeroing from the last bar if 15:55 bar is absent. Vectorized.
    """
    pos = positions.copy()
    # Minutes since midnight: avoids Python datetime objects per bar
    bar_minutes = index.hour * 60 + index.minute  # numpy int array
    eos_minutes = 15 * 60 + 55                     # 955
    dates_int = index.normalize().view(np.int64)

    is_new = np.empty(len(index), dtype=bool)
    is_new[0] = True
    is_new[1:] = dates_int[1:] != dates_int[:-1]
    session_starts = np.where(is_new)[0]
    session_ends   = np.append(session_starts[1:] - 1, len(index) - 1)

    for s, e in zip(session_starts, session_ends):
        # Find first bar >= 15:55 in this session
        eos_candidates = np.where(bar_minutes[s:e+1] >= eos_minutes)[0]
        start = s + (eos_candidates[0] if len(eos_candidates) else (e - s))
        pos[start: e + 1] = 0
    return pos


# ---------------------------------------------------------------------------
# Core pair execution
# ---------------------------------------------------------------------------

def run_pair(
    log_a_1min: np.ndarray,
    log_b_1min: np.ndarray,
    close_a_1min: np.ndarray,
    close_b_1min: np.ndarray,
    index_1min: pd.DatetimeIndex,
    alpha_pca: float,
    beta_pca: float,
    R: float,
    delta: float,
    half_life_days: float,
    entry_z: float = 2.0,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    rebalance_threshold: float = _REBALANCE_THRESHOLD,
    rebalance_dead_band: float = _REBALANCE_DEAD_BAND,
    eos_flatten: bool = True,
    spread_mean_form: float | None = None,
    spread_std_form: float | None = None,
    # V3.0 ADD — Z entry strategy:
    z_strategy: str = "fixed",
    hl_median_fold: float | None = None,
) -> pd.DataFrame:
    """
    V3.0: Static spread + Z-score + state machine for one pair over one trading window.

    Z entry threshold: per-bar, from build_z_entry_array(strategy, ...).
        - 'fixed': constant `entry_z` (V2.0-equivalent)
        - 'vol':   z_base * max(1, sigma_current/sigma_formation)
        - 'hl':    z_base * sqrt(hl_pair / hl_median_fold)

    Returns bar-level DataFrame:
      [spread, zscore, signal, position, beta_post, z_entry_used,
       rebalance_event, rebalance_cost, rebalance_dshares, rebalance_price]
    """
    n = len(log_a_1min)
    if n < 10:
        return pd.DataFrame()

    # 1. V3.0 F4: static spread (was Kalman in V2.0)
    spread = compute_static_spread(log_a_1min, log_b_1min, alpha_pca, beta_pca)
    # beta_post is constant under V3.0 static spread (rebalance branch will not fire)
    beta_post = np.full(n, beta_pca, dtype=np.float64)

    # 2. Z-score: prefer formation-locked normalization
    if spread_mean_form is not None and spread_std_form is not None and spread_std_form > 1e-10:
        zscore_raw = (spread - spread_mean_form) / spread_std_form
    else:
        z_window = min(int(half_life_days * _BARS_PER_DAY_1MIN), _Z_WINDOW_CAP)
        z_window = max(z_window, 10)
        zscore_raw = rolling_zscore(pd.Series(spread), window=z_window).values

    # 3. Session warmup → NaN
    warmup_mask = _session_warmup_mask(index_1min, _SESSION_WARMUP_BARS)
    zscore_raw[warmup_mask] = np.nan

    # 4. V3.0 A1: kill-zone block-entry mask
    block_entry = build_block_entry_mask(index_1min)

    # 5. V3.0 F2: build per-bar Z entry threshold array
    z_entry_array = build_z_entry_array(
        strategy=z_strategy,
        z_base=entry_z,
        n_bars=n,
        spread=spread if z_strategy == "vol" else None,
        sigma_formation=spread_std_form if z_strategy == "vol" else None,
        hl_pair_days=half_life_days if z_strategy == "hl" else None,
        hl_median_fold=hl_median_fold if z_strategy == "hl" else None,
    )

    # 6. State machine (with V3.0 A1 mask, A2 hard SL, A3 Z-velocity gate)
    signal_raw = _state_machine(zscore_raw, z_entry_array, block_entry)

    # 5. EOS flatten
    if eos_flatten:
        signal_raw = _apply_eos_flatten(signal_raw, index_1min)

    # 6. Execution lag (1 bar) — genuine OOS
    position = np.zeros(n, dtype=np.int8)
    position[1:] = signal_raw[:-1]

    # 7. V3.0 A4: vol-target sizing (was flat $20k/pair in V2.0)
    if n >= _BARS_PER_DAY_1MIN:
        per_pair_dollar = compute_vol_target_notional(spread)
    else:
        # Trading window too short for vol-target — fall back to V2.0 flat sizing
        per_pair_dollar = total_capital / n_open_pairs_max
    short_notional = per_pair_dollar * 0.5

    rebalance_events = np.zeros(n, dtype=bool)
    rebalance_costs  = np.zeros(n, dtype=float)
    rebalance_dshares = np.zeros(n, dtype=float)   # signed delta-shares of B leg
    rebalance_price   = np.zeros(n, dtype=float)   # price of B at rebalance bar
    beta_ref = beta_pca
    in_deadband = False

    for i in range(n):
        if position[i] == 0:
            beta_ref = beta_post[i]  # reset ref when flat
            in_deadband = False
            continue

        drift = (beta_post[i] - beta_ref) / abs(beta_ref) if abs(beta_ref) > 1e-10 else 0.0

        if in_deadband and abs(drift) < rebalance_dead_band:
            continue  # still inside dead band

        in_deadband = False
        if abs(drift) > rebalance_threshold:
            # PnL parameterises shares_b = (short_notional * beta) / p_b (signed).
            # On a beta change delta_beta, the share delta needed is
            #   delta_shares_b = (short_notional / p_b) * delta_beta   (signed)
            delta_beta     = beta_post[i] - beta_ref
            p_b_reb        = max(close_b_1min[i], 1e-6)
            d_shares_b     = (short_notional * delta_beta) / p_b_reb   # signed
            cost           = _REBALANCE_COST_BPS * abs(d_shares_b) * p_b_reb
            rebalance_events[i]  = True
            rebalance_costs[i]   = cost
            rebalance_dshares[i] = d_shares_b
            rebalance_price[i]   = p_b_reb
            beta_ref    = beta_post[i]
            in_deadband = True  # enter dead band after rebalance

    df = pd.DataFrame({
        "spread":           spread,
        "spread_prior":     spread,   # alias for backward compat with phase3_backtest.metrics_runner (V2 Kalman naming)
        "zscore":           zscore_raw,
        "z_entry_used":     z_entry_array,
        "signal":           signal_raw,
        "position":         position,
        "beta_post":        beta_post,
        "rebalance_event":  rebalance_events,
        "rebalance_cost":   rebalance_costs,
        "rebalance_dshares": rebalance_dshares,
        "rebalance_price":   rebalance_price,
    }, index=index_1min)

    return df


# ---------------------------------------------------------------------------
# Max-holding cap — apply after execution, before PnL
# ---------------------------------------------------------------------------

def _apply_max_holding(pair_df: pd.DataFrame, max_holding_bars: int) -> pd.DataFrame:
    """
    Post-process an execution DataFrame to force position to 0 when it has
    been continuously non-zero for more than max_holding_bars bars.
    Modifies only the 'position' and 'signal' columns in-place (copy returned).
    """
    df = pair_df.copy()
    pos = df["position"].values.copy()
    sig = df["signal"].values.copy()
    forced = np.zeros(len(pos), dtype=bool)
    n = len(pos)
    hold_count = 0
    for i in range(n):
        if pos[i] != 0:
            hold_count += 1
            if hold_count > max_holding_bars:
                pos[i] = 0
                forced[i] = True
                # Maintain position[i] == signal[i-1] invariant so the
                # lookahead check in metrics_runner doesn't false-alarm.
                if i > 0:
                    sig[i - 1] = 0
        else:
            hold_count = 0
    df["position"]     = pos
    df["signal"]       = sig   # sig[i-1] zeroed for each forced exit to maintain invariant
    df["forced_exit"]  = forced
    return df


# ---------------------------------------------------------------------------
# Fold-level runner — called by Phase 4 orchestrator
# ---------------------------------------------------------------------------

def run_fold_execution(
    pairs_df: pd.DataFrame,
    trading_1min: dict[str, pd.DataFrame],
    delta: float,
    entry_z: float = 2.0,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    eos_flatten: bool = True,
    formation_5min: dict | None = None,
    formation_ref: dict | None = None,
    # V3.0 ADD — Z entry strategy parameters
    z_strategy: str = "fixed",
) -> dict[str, pd.DataFrame]:
    """
    V3.0: run execution for all pairs in a fold.

    z_strategy: "fixed" | "vol" | "hl" — see engine/z_strategies.py
    `delta` parameter is retained for V2-API compatibility but ignored
    under V3.0 static spread.
    """
    # Apply per-ticker concentration cap
    pairs_capped = apply_ticker_concentration_cap(pairs_df)
    log.info("Fold execution V3.0: %d pairs after concentration cap, z_strategy=%s",
             len(pairs_capped), z_strategy)

    # V3.0 HL-conditional Z needs fold-median half-life
    hl_median_fold: float | None = None
    if z_strategy == "hl" and "half_life_days" in pairs_capped.columns and len(pairs_capped) > 0:
        hl_median_fold = float(pairs_capped["half_life_days"].median())

    results: dict[tuple[str, str], pd.DataFrame] = {}

    for _, row in pairs_capped.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        if ta not in trading_1min or tb not in trading_1min:
            log.debug("Skip pair %s/%s: 1-min data missing", ta, tb)
            continue

        df_a = trading_1min[ta].copy()
        df_b = trading_1min[tb].copy()

        # Compute log_close if not pre-computed (1min parquet stores raw OHLCV)
        if "log_close" not in df_a.columns:
            df_a["log_close"] = np.log(df_a["close"].clip(lower=1e-10))
        if "log_close" not in df_b.columns:
            df_b["log_close"] = np.log(df_b["close"].clip(lower=1e-10))

        # Align on common timestamps
        df_ab = df_a[["log_close", "close"]].join(
            df_b[["log_close", "close"]], how="inner", lsuffix="_a", rsuffix="_b"
        ).dropna()

        if len(df_ab) < 20:
            continue

        # V3.0 F4: formation-window spread mean/std via static spread
        # (was: run_kalman on formation). Under V3.0 static, this is direct mean/std.
        _form_src = formation_ref if formation_ref is not None else formation_5min
        spread_mean_form = spread_std_form = None
        if _form_src is not None:
            fa = _form_src.get(ta)
            fb = _form_src.get(tb)
            if (fa is not None and fb is not None
                    and "log_close" in fa.columns and "log_close" in fb.columns):
                df_form = (
                    fa[["log_close"]].rename(columns={"log_close": "lc_a"})
                    .join(fb[["log_close"]].rename(columns={"log_close": "lc_b"}), how="inner")
                    .dropna()
                )
                if len(df_form) > 20:
                    try:
                        spread_mean_form, spread_std_form = compute_formation_z_params(
                            df_form["lc_a"].values, df_form["lc_b"].values,
                            float(row["alpha_pca"]), float(row["beta_pca"]),
                        )
                    except (ValueError, AssertionError) as e:
                        log.debug("formation_z_params skipped for %s/%s: %s", ta, tb, e)

        pair_df = run_pair(
            log_a_1min   = df_ab["log_close_a"].values,
            log_b_1min   = df_ab["log_close_b"].values,
            close_a_1min = df_ab["close_a"].values,
            close_b_1min = df_ab["close_b"].values,
            index_1min   = df_ab.index,
            alpha_pca    = float(row["alpha_pca"]),
            beta_pca     = float(row["beta_pca"]),
            R            = float(row["R_measurement_noise"]),
            delta        = delta,
            half_life_days = float(row["half_life_days"]),
            entry_z      = entry_z,
            n_open_pairs_max = n_open_pairs_max,
            total_capital    = total_capital,
            eos_flatten      = eos_flatten,
            spread_mean_form = spread_mean_form,
            spread_std_form  = spread_std_form,
            z_strategy       = z_strategy,
            hl_median_fold   = hl_median_fold,
        )

        if not pair_df.empty:
            results[(ta, tb)] = pair_df

    log.info("Fold execution V3.0 complete: %d pairs ran successfully", len(results))
    return results
