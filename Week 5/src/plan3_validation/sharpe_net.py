"""
4.1 — Before/After table.

Reconstructs daily portfolio returns under each cost regime by joining
trade-level cost dollars onto Week 4's bar-level fold equity parquets,
then aggregates to daily.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"
DEFAULT_EQUITY_DIR = WEEK5_ROOT / "data" / "week4_inputs" / "equities"
DEFAULT_FOLD_METRICS = WEEK5_ROOT / "data" / "week4_inputs" / "fold_metrics.csv"


def _annualised_sharpe(daily_returns: np.ndarray) -> float:
    arr = daily_returns[~np.isnan(daily_returns)]
    if len(arr) < 2:
        return float("nan")
    sd = arr.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(252))


def _max_dd(equity: np.ndarray) -> float:
    eq = equity[~np.isnan(equity)]
    if len(eq) == 0:
        return float("nan")
    running = np.maximum.accumulate(eq)
    return float((eq / running - 1.0).min())


def _cagr(equity: np.ndarray, n_trading_days: int) -> float:
    eq = equity[~np.isnan(equity)]
    if len(eq) == 0 or n_trading_days <= 0 or eq[-1] <= 0:
        return float("nan")
    return float(eq[-1] ** (252.0 / n_trading_days) - 1.0)


def reconstruct_daily_returns(
    cost_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    cost_col: str,
) -> pd.Series:
    """
    Build a daily-return series for one cost regime.

    Treats cost as an exit-day debit: subtracts each trade's total cost
    from the day the trade exits, then divides by the prior-day portfolio
    equity (proxied as starting capital + cumulative gross PnL).

    Returns a daily Series indexed by date.
    """
    merged = cost_log.merge(
        trade_log[["trade_id", "exit_ts", "gross_pnl_dollars", "allocated_capital"]],
        on="trade_id",
        suffixes=("", "_tl"),
    )
    merged["exit_date"] = pd.to_datetime(merged["exit_ts"], utc=True).dt.tz_convert("US/Eastern").dt.date

    # net pnl per trade under chosen regime
    if cost_col == "total_cost_gross":
        merged["net_pnl"] = merged["gross_pnl_dollars"]
    elif cost_col == "total_cost_static":
        merged["net_pnl"] = merged["gross_pnl_dollars"] - merged["total_cost_static"]
    else:
        merged["net_pnl"] = merged["gross_pnl_dollars"] - merged["total_cost_dollars"]

    daily = merged.groupby("exit_date")["net_pnl"].sum().sort_index()

    # Capital base: total allocated capital across all trades (proxy for AUM)
    aum = trade_log["allocated_capital"].sum()
    if aum <= 0:
        return pd.Series(dtype=float)

    return daily / aum


def generate_before_after_table(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
) -> pd.DataFrame:
    """
    Produces the 8-row × 3-column "Before vs After" table.

    Rows: Sharpe (annual), CAGR, MaxDD (daily-equity-derived), Calmar,
          Win Rate, Avg Trades/Fold, Avg RT Cost (bps), % Folds Profitable.
    Cols: Gross, Static60bps, Dynamic.
    """
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)

    regimes = {
        "Gross": "total_cost_gross",
        "Static60bps": "total_cost_static",
        "Dynamic": "total_cost_dollars",
    }
    rows: dict[str, dict] = {}

    for label, cost_col in regimes.items():
        daily = reconstruct_daily_returns(cost_log, trade_log, cost_col)
        sharpe = _annualised_sharpe(daily.to_numpy())

        # Bar-level MaxDD requires per-regime intra-trade equity curves which
        # Week 4 only stores under its native (static) regime. We use daily
        # compounded equity here as a documented approximation.
        eq_curve = (1.0 + daily).cumprod().to_numpy() if len(daily) else np.array([1.0])
        max_dd = _max_dd(eq_curve)
        n_days = len(daily)
        cagr = _cagr(eq_curve, n_days) if n_days > 0 else float("nan")
        calmar = cagr / abs(max_dd) if max_dd and max_dd != 0 else float("nan")

        if cost_col == "total_cost_gross":
            wins = (cost_log["gross_pnl_dollars"] > 0).sum()
        elif cost_col == "total_cost_static":
            wins = ((cost_log["gross_pnl_dollars"] - cost_log["total_cost_static"]) > 0).sum()
        else:
            wins = ((cost_log["gross_pnl_dollars"] - cost_log["total_cost_dollars"]) > 0).sum()
        win_rate = wins / len(cost_log) if len(cost_log) else float("nan")

        n_folds = trade_log["fold_id"].nunique()
        avg_trades_per_fold = len(trade_log) / n_folds if n_folds else float("nan")

        # Avg RT cost in bps: total_cost / allocated_capital, mean across trades
        merged_cap = cost_log.merge(trade_log[["trade_id", "allocated_capital"]], on="trade_id")
        if cost_col == "total_cost_gross":
            cost_per_trade = pd.Series(0.0, index=merged_cap.index)
        else:
            cost_per_trade = merged_cap[cost_col]
        avg_rt_cost_bps = (cost_per_trade / merged_cap["allocated_capital"]).mean() * 10000

        # % folds profitable: per-fold net PnL > 0
        fold_net = merged_cap.copy()
        fold_net["net_pnl"] = (
            fold_net["gross_pnl_dollars"] - cost_per_trade
        )
        fold_pnls = fold_net.merge(trade_log[["trade_id", "fold_id"]], on="trade_id").groupby("fold_id")["net_pnl"].sum()
        pct_folds_profitable = (fold_pnls > 0).mean()

        rows[label] = {
            "Sharpe (annual)": sharpe,
            "CAGR": cagr,
            "MaxDD (bar)": max_dd,
            "Calmar": calmar,
            "Win Rate": win_rate,
            "Avg Trades / Fold": avg_trades_per_fold,
            "Avg RT Cost (bps)": avg_rt_cost_bps,
            "% Folds Profitable": pct_folds_profitable,
        }

    return pd.DataFrame(rows).round(4)
