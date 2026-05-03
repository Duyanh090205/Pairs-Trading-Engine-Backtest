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
