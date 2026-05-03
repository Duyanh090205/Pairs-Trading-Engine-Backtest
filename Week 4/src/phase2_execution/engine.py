"""
Phase 2 — Execution Engine

Per fold, per pair:
  1. apply_ticker_concentration_cap  — portfolio construction filter (max 5 pairs/ticker)
  2. run_kalman on 1-min trading data → prior spread series
  3. rolling Z-score with session warmup
  4. Numba state machine → raw positions
  5. 1-bar execution lag → executed positions
  6. Position sizing (dollar-neutral)
  7. Threshold rebalance with hysteresis (default X=10%)

Returns bar-level trade log per pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

from src.phase2_execution.kalman import run_kalman
from src.utils.stats import rolling_zscore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BARS_PER_DAY_1MIN = 390
_SESSION_WARMUP_BARS = 30       # first 30 bars each session → NaN zscore
_Z_WINDOW_CAP = 2000            # cap on rolling Z window in bars
_REBALANCE_THRESHOLD = 0.10     # 10% beta drift triggers rebalance
_REBALANCE_DEAD_BAND = 0.05     # 5% = X/2, hysteresis dead band
_REBALANCE_COST_BPS = 0.0030    # 30 bps one-side on delta shares
_N_OPEN_PAIRS_MAX = 50          # default portfolio cap
_TOTAL_CAPITAL = 1_000_000.0    # $1M notional (normalised)
_MAX_PAIRS_PER_TICKER = 5       # concentration cap


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
# Numba state machine
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _state_machine(
    zscores: np.ndarray,
    entry_z: float,
) -> np.ndarray:
    """
    Pure signal: position array from Z-score series.
    state +1 = long A short B, -1 = short A long B, 0 = flat.
    Entry on Z crossing +/- entry_z, exit on Z zero-crossing.
    NaN Z: hold current state, no new entry.
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    state = np.int8(0)

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            positions[i] = state
            continue

        if state == 0:
            if z < -entry_z:
                state = np.int8(1)
            elif z > entry_z:
                state = np.int8(-1)
        else:
            # Exit on zero-crossing
            if (state == 1 and z >= 0.0) or (state == -1 and z <= 0.0):
                state = np.int8(0)

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
) -> pd.DataFrame:
    """
    Run Kalman + Z-score + state machine for one pair over one trading window.

    Returns bar-level DataFrame with columns:
      [spread_prior, zscore, signal, position, beta_post,
       rebalance_event, rebalance_cost]
    """
    n = len(log_a_1min)
    if n < 10:
        return pd.DataFrame()

    # 1. Kalman on 1-min data
    spread_prior, _, _, _, beta_post = run_kalman(
        log_a_1min, log_b_1min, alpha_pca, beta_pca, R, delta,
    )

    # 2. Z-score: use formation-window-locked mean/std when available (eliminates
    # rolling-mean contamination that inverts EOS vs zero-cross PnL sign).
    # Fall back to rolling Z only when formation stats are absent.
    if spread_mean_form is not None and spread_std_form is not None and spread_std_form > 1e-10:
        zscore_raw = (spread_prior - spread_mean_form) / spread_std_form
    else:
        z_window = min(int(half_life_days * _BARS_PER_DAY_1MIN), _Z_WINDOW_CAP)
        z_window = max(z_window, 10)
        zscore_raw = rolling_zscore(pd.Series(spread_prior), window=z_window).values

    # 3. Session warmup → NaN
    warmup_mask = _session_warmup_mask(index_1min, _SESSION_WARMUP_BARS)
    zscore_raw[warmup_mask] = np.nan

    # 4. State machine
    signal_raw = _state_machine(zscore_raw, entry_z)

    # 5. EOS flatten
    if eos_flatten:
        signal_raw = _apply_eos_flatten(signal_raw, index_1min)

    # 6. Execution lag (1 bar) — genuine OOS
    position = np.zeros(n, dtype=np.int8)
    position[1:] = signal_raw[:-1]

    # 7. Threshold rebalance with hysteresis
    per_pair_dollar = total_capital / n_open_pairs_max
    short_notional  = per_pair_dollar * 0.5

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
        "spread_prior":     spread_prior,
        "zscore":           zscore_raw,
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
) -> dict[str, pd.DataFrame]:
    """
    Run execution for all pairs in a fold.

    Parameters
    ----------
    pairs_df       : Phase 1 output (post concentration cap applied here)
    trading_1min   : dict ticker -> 1-min DataFrame with 'log_close' and 'close'
    delta          : selected by delta_selector for this fold
    entry_z        : Z-score entry threshold

    Returns
    -------
    dict (ticker_a, ticker_b) -> bar-level execution DataFrame
    """
    # Apply per-ticker concentration cap
    pairs_capped = apply_ticker_concentration_cap(pairs_df)
    log.info("Fold execution: %d pairs after concentration cap", len(pairs_capped))

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

        # Compute formation-window spread mean/std for locked Z-score normalization.
        # formation_ref (1-min) is preferred — same frequency as trading window.
        # formation_5min is accepted as fallback (approximate for near-static delta).
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
                        sp_form, *_ = run_kalman(
                            df_form["lc_a"].values, df_form["lc_b"].values,
                            float(row["alpha_pca"]), float(row["beta_pca"]),
                            float(row["R_measurement_noise"]), delta,
                        )
                        sp_clean = sp_form[~np.isnan(sp_form)]
                        if len(sp_clean) > 20:
                            spread_mean_form = float(np.mean(sp_clean))
                            spread_std_form  = max(float(np.std(sp_clean)), 1e-10)
                    except Exception:
                        pass

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
        )

        if not pair_df.empty:
            results[(ta, tb)] = pair_df

    log.info("Fold execution complete: %d pairs ran successfully", len(results))
    return results
