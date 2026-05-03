"""
Phase 3 — Bar-Level PnL Engine

Consumes Phase 2 engine output (pair_df) and computes:
  - Bar-level gross and net PnL
  - Cost decomposition: commission | borrow | rebalance
  - Trade log with entry/exit timestamps and reasons
  - Aggregate metrics: Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate

Interface contract from Phase 2:
  pair_df columns used:
    position      int8   Already 1-bar lagged; +1=long A/short B, -1=short A/long B
    signal        int8   Pre-lag signal (timestamp verification only)
    beta_post     f64    Read at entry bar to size shares_B
    rebalance_cost f64   Pre-computed dollars; add directly to cost decomp
    spread_prior  f64    For Kalman degenerate check

Entry: position[t] != 0 and position[t-1] == 0  (use prices[t] to size)
Exit:  position[t] == 0 and position[t-1] != 0  (charge TC at prices[t])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_TC_BPS_DEFAULT      = 30.0   # one-way bps per leg
_BORROW_BPS_YR       = 50.0   # annual borrow rate on short leg


# ---------------------------------------------------------------------------
# Per-pair PnL
# ---------------------------------------------------------------------------

def compute_pair_pnl(
    pair_df: pd.DataFrame,
    close_a: pd.Series,
    close_b: pd.Series,
    per_pair_dollar: float,
    tc_bps: float = _TC_BPS_DEFAULT,
    borrow_bps_yr: float = _BORROW_BPS_YR,
) -> dict:
    """
    Compute bar-level PnL for a single pair.

    Parameters
    ----------
    pair_df        : engine output DataFrame (position, signal, beta_post, rebalance_cost)
    close_a/b      : 1-min close price Series aligned to pair_df.index
    per_pair_dollar: total_capital / n_open_pairs_max
    tc_bps         : one-way TC in basis points (30 bps entry + 30 bps exit)
    borrow_bps_yr  : annual borrow rate in bps on short leg notional

    Returns dict with keys:
      bar_pnl, bar_gross, trade_log,
      cost_commission, cost_borrow, cost_rebalance
    """
    n = len(pair_df)
    if n < 2:
        return _empty_pair_pnl(pair_df.index if n else None)

    # Align prices to pair_df index (inner join already done in engine)
    ca = close_a.reindex(pair_df.index).ffill().values
    cb = close_b.reindex(pair_df.index).ffill().values

    tc_rate      = tc_bps / 10_000.0
    borrow_daily = (borrow_bps_yr / 10_000.0) / 252.0
    long_notional  = per_pair_dollar * 0.5
    short_notional = per_pair_dollar * 0.5

    pos      = pair_df["position"].values.astype(np.int8)
    beta     = pair_df["beta_post"].values
    reb_cost = pair_df["rebalance_cost"].values
    # Engine-side rebalance metadata (signed delta-shares + price at rebalance bar).
    # Older engine outputs lack these columns; default to zeros for backward compat.
    reb_dshares = (pair_df["rebalance_dshares"].values
                   if "rebalance_dshares" in pair_df.columns else np.zeros(n, dtype=float))
    reb_price   = (pair_df["rebalance_price"].values
                   if "rebalance_price" in pair_df.columns else np.zeros(n, dtype=float))
    idx      = pair_df.index
    # forced_exit column present when _apply_max_holding was used
    forced_exit = pair_df["forced_exit"].values if "forced_exit" in pair_df.columns else np.zeros(n, dtype=bool)

    bar_pnl   = np.zeros(n)
    bar_gross = np.zeros(n)
    cost_commission = 0.0
    cost_borrow     = 0.0

    trade_log  = []
    rebalance_log = []   # per-event records for Week-5 hand-off
    shares_a   = 0.0
    shares_b   = 0.0
    direction  = 0
    entry_bar  = -1
    entry_ts   = None
    prev_date  = None
    # Entry-bar prices retained for entry-notional records and exit comparison
    entry_p_a  = 0.0
    entry_p_b  = 0.0
    # In-flight trade index (incremented on each entry; rebalance events tag
    # this id so the consumer can map a rebalance to its enclosing trade).
    pair_trade_idx = 0

    for i in range(1, n):
        curr = int(pos[i])
        prev = int(pos[i - 1])
        curr_date = idx[i].date()

        # ---- REBALANCE COST (pre-computed by engine) ----
        # Apply BEFORE entry/exit so the trade-level net_pnl summation
        # at exit captures any rebalance cost incurred on the exit bar.
        bar_pnl[i] -= reb_cost[i]
        if reb_cost[i] > 0 and direction != 0:
            # Engine emits signed delta_shares and price_at_rebalance per event.
            # Notional rebalanced = |delta_shares| * price_at_rebalance.
            d_shares = float(reb_dshares[i])
            p_reb    = float(reb_price[i])
            notional_rebalanced = abs(d_shares) * p_reb
            rebalance_log.append({
                "pair_trade_idx":      pair_trade_idx,   # in-flight trade index for this pair
                "rebalance_ts":        idx[i],
                "delta_shares":        d_shares,         # signed
                "price_at_rebalance":  p_reb,
                "notional_rebalanced": notional_rebalanced,
                "cost_dollars":        float(reb_cost[i]),
            })

        # ---- ENTRY ----
        if curr != 0 and prev == 0:
            direction = curr
            # Use bar i-1 prices: position[i] was triggered by signal[i-1],
            # so execution fills at the close of bar i-1 (the signal bar).
            p_a = max(ca[i - 1], 1e-6)
            p_b = max(cb[i - 1], 1e-6)
            beta_entry = float(beta[i - 1])
            beta_sign  = 1.0 if beta_entry >= 0 else -1.0  # cointegration sign of B leg
            # shares_a always positive; shares_b is the *signed* hedge amount.
            # For beta>=0: long A / short B (per direction); shares_b > 0.
            # For beta<0:  long A / long  B (per direction); shares_b < 0.
            # Spread = ln(A) - alpha - beta*ln(B); reversion captured by
            # gross = direction * (shares_a*dp_a - shares_b*dp_b) with signed shares_b.
            shares_a = long_notional / p_a
            shares_b = (short_notional * beta_entry) / p_b   # signed
            # TC on absolute notional traded on each leg
            tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
            bar_pnl[i] -= tc
            cost_commission += tc
            entry_bar = i
            entry_ts  = idx[i]
            entry_p_a = p_a
            entry_p_b = p_b
            pair_trade_idx += 1
            prev_date = curr_date

        # ---- GROSS PnL while in position ----
        if curr != 0:
            dp_a = ca[i] - ca[i - 1]
            dp_b = cb[i] - cb[i - 1]
            gross = direction * (shares_a * dp_a - shares_b * dp_b)
            bar_gross[i] = gross
            bar_pnl[i] += gross

        # ---- BORROW COST (daily accrual on the short leg only) ----
        # Determine which leg is short given (direction, sign(beta)):
        #   direction>0, beta>=0:  A long,  B short  -> short notional = |shares_b|*p
        #   direction>0, beta<0 :  A long,  B long   -> no short leg
        #   direction<0, beta>=0:  A short, B long   -> short notional = shares_a*p
        #   direction<0, beta<0 :  A short, B short  -> short notional = both
        if curr != 0 and prev_date is not None and curr_date != prev_date:
            short_notional_today = 0.0
            if direction > 0 and beta_sign > 0:
                short_notional_today = abs(shares_b) * max(cb[i - 1], 1e-6)
            elif direction < 0 and beta_sign > 0:
                short_notional_today = shares_a * max(ca[i - 1], 1e-6)
            elif direction < 0 and beta_sign < 0:
                short_notional_today = (shares_a * max(ca[i - 1], 1e-6)
                                        + abs(shares_b) * max(cb[i - 1], 1e-6))
            # direction>0, beta<0: both legs long, no borrow
            borrow = borrow_daily * short_notional_today
            bar_pnl[i] -= borrow
            cost_borrow += borrow

        if curr != 0:
            prev_date = curr_date

        # ---- EXIT ----
        if curr == 0 and prev != 0:
            # Same logic as entry: position[i]=0 means signal[i-1] triggered exit,
            # so execution fills at close of bar i-1.
            p_a = max(ca[i - 1], 1e-6)
            p_b = max(cb[i - 1], 1e-6)
            tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
            bar_pnl[i] -= tc
            cost_commission += tc

            # Determine exit reason
            if forced_exit[i]:
                exit_reason = "max_holding"
            else:
                exit_reason = _exit_reason(idx[i])
            trade_net_pnl = bar_pnl[entry_bar: i + 1].sum()
            trade_gross_pnl = bar_gross[entry_bar: i + 1].sum()

            # Notionals at entry/exit (positive dollar amounts traded on each leg).
            # shares_b is signed (cointegration sign of B leg); the notional reported
            # is the absolute dollar amount traded, regardless of long/short direction.
            notional_a_entry = float(shares_a * entry_p_a)
            notional_b_entry = float(abs(shares_b) * entry_p_b)
            notional_a_exit  = float(shares_a * p_a)
            notional_b_exit  = float(abs(shares_b) * p_b)

            # side_A = direction (long when +1, short when -1)
            # side_B = -direction * sign(beta) — derived from cointegration sign:
            #   beta>=0: B is opposite of A  -> side_B = -direction
            #   beta<0 : B is same  as A     -> side_B = +direction
            side_a = int(direction)
            side_b = int(-direction * beta_sign)
            trade_log.append({
                "pair_trade_idx":   pair_trade_idx,        # local index — runner promotes to global trade_id
                "entry_ts":         entry_ts,
                "exit_ts":          idx[i],
                "direction":        direction,
                "side_A":           side_a,
                "side_B":           side_b,
                "n_bars":           i - entry_bar,
                "gross_pnl":        float(trade_gross_pnl),
                "net_pnl":          float(trade_net_pnl),
                "gross_bps":        float(trade_gross_pnl / per_pair_dollar * 10_000),
                "net_bps":          float(trade_net_pnl / per_pair_dollar * 10_000),
                "exit_reason":      exit_reason,
                "notional_a_entry": notional_a_entry,
                "notional_b_entry": notional_b_entry,
                "notional_a_exit":  notional_a_exit,
                "notional_b_exit":  notional_b_exit,
                "allocated_capital": float(per_pair_dollar),
            })
            shares_a  = 0.0
            shares_b  = 0.0
            direction = 0
            entry_p_a = 0.0
            entry_p_b = 0.0
            prev_date = None

    # Handle position still open at window end (carry-forward in orchestrator)
    if direction != 0 and shares_a > 0:
        p_a = max(ca[-1], 1e-6)
        p_b = max(cb[-1], 1e-6)
        tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
        bar_pnl[-1] -= tc
        cost_commission += tc
        trade_net_pnl   = bar_pnl[entry_bar:].sum()
        trade_gross_pnl = bar_gross[entry_bar:].sum()
        notional_a_entry = float(shares_a * entry_p_a)
        notional_b_entry = float(abs(shares_b) * entry_p_b)
        notional_a_exit  = float(shares_a * p_a)
        notional_b_exit  = float(abs(shares_b) * p_b)
        side_a = int(direction)
        side_b = int(-direction * beta_sign)
        trade_log.append({
            "pair_trade_idx":   pair_trade_idx,
            "entry_ts":         entry_ts,
            "exit_ts":          idx[-1],
            "direction":        direction,
            "side_A":           side_a,
            "side_B":           side_b,
            "n_bars":           n - entry_bar,
            "gross_pnl":        float(trade_gross_pnl),
            "net_pnl":          float(trade_net_pnl),
            "gross_bps":        float(trade_gross_pnl / per_pair_dollar * 10_000),
            "net_bps":          float(trade_net_pnl / per_pair_dollar * 10_000),
            "exit_reason":      "end_of_window",
            "notional_a_entry": notional_a_entry,
            "notional_b_entry": notional_b_entry,
            "notional_a_exit":  notional_a_exit,
            "notional_b_exit":  notional_b_exit,
            "allocated_capital": float(per_pair_dollar),
        })

    cost_rebalance = float(reb_cost.sum())

    return {
        "bar_pnl":          pd.Series(bar_pnl, index=idx),
        "bar_gross":        pd.Series(bar_gross, index=idx),
        "trade_log":        trade_log,
        "rebalance_log":    rebalance_log,   # list of dicts; runner promotes pair_trade_idx -> trade_id
        "cost_commission":  cost_commission,
        "cost_borrow":      cost_borrow,
        "cost_rebalance":   cost_rebalance,
    }


def _exit_reason(exit_ts: pd.Timestamp) -> str:
    import datetime
    if exit_ts.time() >= datetime.time(15, 56):
        return "eos"
    return "zero_cross"


def _empty_pair_pnl(index) -> dict:
    empty = pd.Series(dtype=float) if index is None else pd.Series(0.0, index=index)
    return {
        "bar_pnl": empty, "bar_gross": empty,
        "trade_log": [], "rebalance_log": [],
        "cost_commission": 0.0,
        "cost_borrow": 0.0, "cost_rebalance": 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    bar_pnl: pd.Series,
    total_capital: float,
    trade_log: list[dict],
) -> dict:
    """
    Compute Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate from bar PnL.

    bar_pnl       : total net PnL in dollars per bar (sum across all pairs)
    total_capital : $1M normalizer
    trade_log     : list of trade dicts from compute_pair_pnl
    """
    if bar_pnl.empty or bar_pnl.abs().sum() == 0:
        return _zero_metrics()

    # Bar-level equity curve starting at 1.0
    # Clamp bar returns to -1 floor so equity never goes negative
    bar_returns  = (bar_pnl / total_capital).clip(lower=-1.0 + 1e-9)
    bar_equity   = (1.0 + bar_returns).cumprod()

    # MaxDD on bar-level equity (captures intraday drawdowns)
    rolling_peak = bar_equity.cummax()
    drawdowns    = bar_equity / rolling_peak - 1.0
    max_dd       = float(drawdowns.min())

    # Daily returns for Sharpe — group by trading day, drop empty days.
    # Previous implementation used resample("D") which created weekend/holiday
    # zero-bins, deflating both mean and std and biasing Sharpe toward zero
    # (and CAGR's calendar-day exponent).
    daily_pnl_full = bar_pnl.groupby(bar_pnl.index.normalize()).sum()
    # Trading day = day with any non-zero bar activity
    bar_active = bar_pnl.abs().groupby(bar_pnl.index.normalize()).sum() > 0
    daily_pnl_trading = daily_pnl_full[bar_active]
    daily_returns = daily_pnl_trading / total_capital
    n_trading_days = len(daily_returns)

    std_daily = float(daily_returns.std(ddof=1)) if n_trading_days > 1 else 0.0
    sharpe    = float(daily_returns.mean() / std_daily * np.sqrt(252)) if std_daily > 1e-12 else 0.0

    # CAGR using TRADING days as the base for annualization
    total_ret = float(bar_equity.iloc[-1]) - 1.0
    cagr      = float((1.0 + total_ret) ** (252.0 / max(n_trading_days, 1)) - 1.0) if total_ret > -1.0 else -1.0

    # Calmar (bar-level MaxDD)
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # Trade stats
    n_trades = len(trade_log)
    if n_trades > 0:
        net_pnls = [t["net_pnl"] for t in trade_log]
        win_rate = float(sum(1 for p in net_pnls if p > 0) / n_trades)
        avg_hold = float(np.mean([t["n_bars"] for t in trade_log]))
        avg_net_bps = float(np.mean([t["net_bps"] for t in trade_log]))
    else:
        win_rate = 0.0
        avg_hold = 0.0
        avg_net_bps = 0.0

    return {
        "sharpe":         sharpe,
        "max_dd":         max_dd,
        "cagr":           cagr,
        "calmar":         calmar,
        "win_rate":       win_rate,
        "n_trades":       n_trades,
        "avg_holding_bars": avg_hold,
        "avg_net_bps":    avg_net_bps,
        "bar_equity":     bar_equity,
        "daily_returns":  daily_returns,
        "total_return":   total_ret,
    }


def _zero_metrics() -> dict:
    return {
        "sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "calmar": 0.0,
        "win_rate": 0.0, "n_trades": 0, "avg_holding_bars": 0.0,
        "avg_net_bps": 0.0, "bar_equity": pd.Series(dtype=float),
        "daily_returns": pd.Series(dtype=float), "total_return": 0.0,
    }
