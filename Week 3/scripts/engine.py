"""
Signal and PnL Engine Module - Week 3
Implements vectorized state machine, PnL calculations,
Version A (OLS) and Version B (Kalman) sizing logic.
"""

import sys
import os
import numpy as np
import pandas as pd

# Week 2 signal engine imports (runtime sys.path injection)
_W2_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Week 2", "week2_signal_engine")
)
if _W2_ROOT not in sys.path:
    sys.path.insert(0, _W2_ROOT)

from src.signals.spread import compute_spread as _w2_compute_spread
from src.signals.zscore import compute_zscore, apply_session_warmup
from src.signals.state_machine import generate_positions, count_trades

EASTERN = "America/New_York"


# ---------------------------------------------------------------------------
# CP5.1 — Spread computation
# ---------------------------------------------------------------------------

def compute_spread(df_a, df_b, alpha, beta):
    """
    Compute log-price spread: S(t) = log(A) - alpha - beta * log(B).

    Parameters
    ----------
    df_a, df_b : pd.DataFrame or pd.Series
        Must have a 'close' column (or be a Series of close prices).
    alpha, beta : float
        OLS hedge ratio parameters (verified by verify_params.py).

    Returns
    -------
    pd.Series — spread values, index = shared timestamps (inner join).
    """
    # Normalise to Series
    close_a = df_a["close"] if isinstance(df_a, pd.DataFrame) else df_a
    close_b = df_b["close"] if isinstance(df_b, pd.DataFrame) else df_b

    # Deduplicate index (H2 backdating can create colliding timestamps)
    # groupby(level=0).last() avoids pandas 2.x duplicate-label restrictions
    if close_a.index.duplicated().any():
        close_a = close_a.groupby(level=0).last()
    if close_b.index.duplicated().any():
        close_b = close_b.groupby(level=0).last()

    # Align on shared index (inner join)
    aligned = pd.DataFrame({"a": close_a, "b": close_b}).dropna()
    with np.errstate(invalid="ignore"):   # log of negative (H4) -> NaN, not error
        spread = (np.log(aligned["a"].values)
                  - alpha
                  - beta * np.log(aligned["b"].values))
    return pd.Series(spread, index=aligned.index, name="spread")


# ---------------------------------------------------------------------------
# CP5.1 — Signal pipeline
# ---------------------------------------------------------------------------

def generate_signals(df_spread, window=680, z_threshold=2.0,
                     warmup=30, use_lag=True):
    """
    Full signal pipeline: Z-score -> session warmup -> state machine -> execution lag.

    H3 detection is column-presence based:
      - pd.DataFrame with 'spread_biased' column -> H3 path (use biased spread)
      - pd.Series -> direct spread input
      - pd.DataFrame without 'spread_biased' -> ValueError

    Parameters
    ----------
    df_spread : pd.Series or pd.DataFrame
    window    : int   — rolling Z-score window (bars)
    z_threshold : float — entry threshold (absolute Z)
    warmup    : int   — bars to suppress after session open
    burn_in   : int   — used implicitly via window//2 in compute_zscore
    use_lag   : bool  — if True, position_executed[t] = position[t-1] (correct)
                        if False, position_executed[t] = position[t] (biased engine, CP6b)
    Note: burn_in (340 bars = window//2) is handled automatically by compute_zscore
          via its default min_periods = window//2. No explicit parameter needed.

    Returns
    -------
    pd.DataFrame with columns:
        spread, rolling_mean, rolling_std, zscore,
        position, signal_valid, position_executed
    """
    # H3 detection
    if isinstance(df_spread, pd.DataFrame):
        if "spread_biased" in df_spread.columns:
            spread_series = df_spread["spread_biased"].dropna()
        else:
            raise ValueError(
                "df_spread is a DataFrame but has no 'spread_biased' column. "
                "Pass a pd.Series for clean/H1/H2/H4 data."
            )
    else:
        spread_series = df_spread

    # Rolling Z-score (min_periods = window//2 = burn_in automatically)
    zdf = compute_zscore(spread_series, window=window)

    # Session warmup: NaN first `warmup` bars of each calendar day
    zdf = apply_session_warmup(zdf, warmup_bars=warmup)

    # State machine (Numba-JIT from Week 2)
    pos = generate_positions(zdf["zscore"], entry_z=z_threshold, exit_z=0.0)

    signal_valid = ~zdf["zscore"].isna()

    if use_lag:
        position_executed = pos.shift(1).fillna(0).astype("int8")
    else:
        position_executed = pos.astype("int8")

    result = pd.DataFrame({
        "spread":            spread_series.reindex(zdf.index),
        "rolling_mean":      zdf["rolling_mean"],
        "rolling_std":       zdf["rolling_std"],
        "zscore":            zdf["zscore"],
        "position":          pos,
        "signal_valid":      signal_valid,
        "position_executed": position_executed,
    })
    return result


# ---------------------------------------------------------------------------
# CP5.2 — Sizing Version A: OLS constant beta
# ---------------------------------------------------------------------------

def sizing_version_a(df_signals, beta=1.0487):
    """
    Compute share counts using constant OLS beta hedge ratio.

    Caller must ensure df_signals has 'close_a' and 'close_b' columns
    (entry leg A and hedge leg B prices) merged in before calling.

    Convention:
      Long  spread (+1): long  1 unit notional leg A, short beta units leg B
      Short spread (-1): short 1 unit notional leg A, long  beta units leg B

    Shares are locked at entry price and forward-filled within each trade.

    Returns df_signals with 'shares_a' and 'shares_b' added.
    """
    return _compute_shares(df_signals, beta_series=None, fixed_beta=beta)


# ---------------------------------------------------------------------------
# CP5.2 — Sizing Version B: Kalman beta monthly rebalance
# ---------------------------------------------------------------------------

def sizing_version_b(df_signals, monthly_beta_map):
    """
    Compute share counts using Kalman beta rebalanced at month boundaries.

    monthly_beta_map : dict {trading_month_int -> beta_float}
      e.g. {7: 0.7646, 8: 0.7737, ..., 12: 0.7788}
      Key = trading month; value = prior month-end Kalman beta.

    Returns df_signals with 'shares_a' and 'shares_b' added.
    """
    # Build a per-bar beta series by mapping each bar's month to the map
    months    = df_signals.index.tz_convert(EASTERN).month
    beta_vals = months.map(monthly_beta_map).astype(float)
    beta_s    = pd.Series(beta_vals, index=df_signals.index)
    return _compute_shares(df_signals, beta_series=beta_s, fixed_beta=None)


def _compute_shares(df_signals, beta_series, fixed_beta):
    """
    Shared logic for Version A and B sizing.
    Uses numpy integer positions throughout — no pandas .loc per entry.
    """
    result  = df_signals.copy()
    pos_arr = result["position_executed"].values
    pa_arr  = result["close_a"].values
    pb_arr  = result["close_b"].values

    prev_pos        = np.empty_like(pos_arr)
    prev_pos[0]     = 0
    prev_pos[1:]    = pos_arr[:-1]
    entry_locs      = np.where((pos_arr != 0) & (prev_pos == 0))[0]
    exit_locs       = np.where((pos_arr == 0) & (prev_pos != 0))[0]

    beta_vals = None if fixed_beta is not None else beta_series.values

    shares_a = np.zeros(len(pos_arr))
    shares_b = np.zeros(len(pos_arr))

    for ep in entry_locs:
        direction = int(pos_arr[ep])
        beta_here = fixed_beta if beta_vals is None else float(beta_vals[ep])
        sa = direction / pa_arr[ep]
        sb = -direction * beta_here / pb_arr[ep]

        # O(log n_exits) lookup instead of O(n_bars) pandas slice
        xi = int(np.searchsorted(exit_locs, ep, side="right"))
        if xi < len(exit_locs):
            xp = int(exit_locs[xi])
            shares_a[ep : xp + 1] = sa
            shares_b[ep : xp + 1] = sb
        else:
            shares_a[ep:] = sa
            shares_b[ep:] = sb

    flat = pos_arr == 0
    shares_a[flat] = 0.0
    shares_b[flat] = 0.0

    result["shares_a"] = shares_a
    result["shares_b"] = shares_b
    return result


# ---------------------------------------------------------------------------
# CP5.3 — PnL engine: Daily mark-to-market
# ---------------------------------------------------------------------------

def compute_daily_pnl(df_positions, entry_bps=30, exit_bps=30):
    """
    Daily mark-to-market PnL engine.

    Parameters
    ----------
    df_positions : pd.DataFrame
        Must contain: position_executed, shares_a, shares_b, close_a, close_b
    entry_bps : int  — cost deducted at entry bar (per round-trip half)
    exit_bps  : int  — cost deducted at exit bar (per round-trip half)

    Returns
    -------
    daily_df : pd.DataFrame
        Per-calendar-day: daily_return, equity, gross_pnl_dollar,
        cost_dollar, net_pnl_dollar, n_entries, n_exits
    trade_log : pd.DataFrame
        Per-trade: signal_ts, exec_ts, side, entry_price_a, entry_price_b,
        exit_price_a, exit_price_b, gross_pnl_bps, entry_cost_bps,
        exit_cost_bps, net_pnl_bps
    """
    df   = df_positions.copy()
    pos  = df["position_executed"]
    sa   = df["shares_a"]
    sb   = df["shares_b"]
    pa   = df["close_a"]
    pb   = df["close_b"]

    # Bar-level price-space PnL
    pnl_a = sa * pa.diff()
    pnl_b = sb * pb.diff()
    bar_pnl = pnl_a.fillna(0) + pnl_b.fillna(0)

    # Detect entry / exit transitions
    is_entry = (pos != 0) & (pos.shift(1).fillna(0) == 0)
    is_exit  = (pos == 0) & (pos.shift(1).fillna(0) != 0)

    # Transaction costs in dollar terms
    # Notional at each bar = |shares_a| * pa + |shares_b| * pb
    notional_bar = sa.abs() * pa + sb.abs() * pb
    prev_notional = notional_bar.shift(1)
    cost = pd.Series(0.0, index=df.index)
    cost[is_entry] -= (entry_bps / 10_000) * notional_bar[is_entry]
    cost[is_exit]  -= (exit_bps  / 10_000) * prev_notional[is_exit]

    # Reference notional for return normalisation
    # At entry, notional_bar ≈ 1.0 + beta (since shares = 1/price and beta/price)
    # Use the first entry's notional as the per-trade reference; average across trades
    entry_notionals = notional_bar[is_entry]
    ref_notional    = entry_notionals.mean() if len(entry_notionals) > 0 else 1.0

    # Build trade log
    trade_log = _build_trade_log(df, pos, pa, pb, bar_pnl, cost,
                                  is_entry, is_exit, ref_notional,
                                  entry_bps, exit_bps)

    # Timestamp verification: exec_ts > signal_ts for all trades
    ts_violations = 0
    for _, row in trade_log.iterrows():
        if pd.notna(row.get("signal_ts")) and pd.notna(row.get("exec_ts")):
            if row["exec_ts"] <= row["signal_ts"]:
                ts_violations += 1
    if ts_violations > 0:
        print(f"  !! RED FLAG: {ts_violations} trades with exec_ts <= signal_ts")

    # Daily aggregation
    df_et     = df.copy()
    df_et["bar_pnl"] = bar_pnl
    df_et["cost"]    = cost
    df_et["date_et"] = df_et.index.tz_convert(EASTERN).normalize()

    daily_grouped = df_et.groupby("date_et")
    daily_gross   = daily_grouped["bar_pnl"].sum()
    daily_cost    = daily_grouped["cost"].sum()
    daily_entries = daily_grouped.apply(lambda g: (
        ((g["position_executed"] != 0) & (g["position_executed"].shift(1).fillna(0) == 0)).sum()
    ), include_groups=False)
    daily_exits   = daily_grouped.apply(lambda g: (
        ((g["position_executed"] == 0) & (g["position_executed"].shift(1).fillna(0) != 0)).sum()
    ), include_groups=False)

    daily_net    = daily_gross + daily_cost
    daily_return = daily_net / ref_notional

    # Equity curve
    equity = (1 + daily_return).cumprod()
    equity.iloc[0] = 1 + daily_return.iloc[0]  # already correct but explicit

    # Sharpe
    daily_df = pd.DataFrame({
        "daily_return":    daily_return,
        "equity":          equity,
        "gross_pnl_dollar": daily_gross,
        "cost_dollar":     daily_cost,
        "net_pnl_dollar":  daily_net,
        "n_entries":       daily_entries,
        "n_exits":         daily_exits,
    })

    return daily_df, trade_log


def _build_trade_log(df, pos, pa, pb, bar_pnl, cost,
                     is_entry, is_exit, ref_notional,
                     entry_bps, exit_bps):
    """Build per-trade log DataFrame.

    Uses cumulative sums so per-trade PnL is O(1) — no per-bar Python loop.
    """
    entry_locs = np.where(is_entry.values)[0]
    exit_locs  = np.where(is_exit.values)[0]
    n_trades   = min(len(entry_locs), len(exit_locs))

    if n_trades == 0:
        return pd.DataFrame()

    # O(n) cumsum once; per-trade slice is then O(1)
    cum_pnl  = np.cumsum(bar_pnl.values)
    cum_cost = np.cumsum(cost.values)
    idx_list = df.index.tolist()
    pos_vals = pos.values
    pa_vals  = pa.values
    pb_vals  = pb.values

    trades = []
    for i in range(n_trades):
        ep   = int(entry_locs[i])
        xp   = int(exit_locs[i])
        prev = ep - 1

        gross_dollar = cum_pnl[xp]  - (cum_pnl[prev]  if prev >= 0 else 0.0)
        cost_dollar  = cum_cost[xp] - (cum_cost[prev] if prev >= 0 else 0.0)
        net_dollar   = gross_dollar + cost_dollar

        # Entry at bar ep's open = bar ep-1's close (1-bar lag model).
        # Exit at bar xp's open = bar xp-1's close.
        # These are the prices that match what the cumsum PnL actually computes.
        trades.append({
            "signal_ts":      idx_list[prev] if prev >= 0 else None,
            "exec_ts":        idx_list[ep],
            "side":           "LONG" if pos_vals[ep] > 0 else "SHORT",
            "entry_price_a":  pa_vals[prev] if prev >= 0 else pa_vals[ep],
            "entry_price_b":  pb_vals[prev] if prev >= 0 else pb_vals[ep],
            "exit_price_a":   pa_vals[xp - 1],
            "exit_price_b":   pb_vals[xp - 1],
            "gross_pnl_bps":  round(gross_dollar / ref_notional * 10_000, 2),
            "entry_cost_bps": round(entry_bps, 2),
            "exit_cost_bps":  round(exit_bps, 2),
            "net_pnl_bps":    round(net_dollar  / ref_notional * 10_000, 2),
        })

    return pd.DataFrame(trades)
