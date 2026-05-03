"""
4.2 — Cost decomposition waterfall.

Per-trade bps decomposition: Gross alpha → Spread → Impact → Borrow → Rebalance → Net.
Partitioned by Bear 2022 (folds 1-6) vs Bull 2023+ (folds 7-45).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"


def _bps(dollars: pd.Series, capital: pd.Series) -> pd.Series:
    return (dollars / capital) * 10_000.0


def generate_cost_waterfall(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ['Overall', 'Bear 2022', 'Bull 2023+']
    with columns:
      n_trades, gross_bps, spread_bps, impact_bps, borrow_bps, rebalance_bps, net_bps.
    Cost columns are reported as negative (consumed alpha).
    """
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)
    df = cost_log.merge(trade_log[["trade_id", "fold_id", "allocated_capital"]], on="trade_id")
    df["regime"] = df["fold_id"].apply(lambda f: "Bear 2022" if f <= 6 else "Bull 2023+")

    rows: list[dict] = []
    for label, sub in [
        ("Overall",    df),
        ("Bear 2022",  df[df["regime"] == "Bear 2022"]),
        ("Bull 2023+", df[df["regime"] == "Bull 2023+"]),
    ]:
        if sub.empty:
            rows.append({
                "regime": label, "n_trades": 0,
                "gross_bps": 0.0, "spread_bps": 0.0, "impact_bps": 0.0,
                "borrow_bps": 0.0, "rebalance_bps": 0.0, "net_bps": 0.0,
            })
            continue
        cap = sub["allocated_capital"]
        gross  = _bps(sub["gross_pnl_dollars"],     cap).mean()
        spread = _bps(sub["spread_cost_dollars"],   cap).mean()
        impact = _bps(sub["impact_cost_dollars"],   cap).mean()
        borrow = _bps(sub["borrow_cost_dollars"],   cap).mean()
        rebal  = _bps(sub["rebalance_cost_dollars"], cap).mean()
        net    = gross - spread - impact - borrow - rebal
        rows.append({
            "regime":        label,
            "n_trades":      len(sub),
            "gross_bps":     round(gross, 2),
            "spread_bps":    round(-spread, 2),
            "impact_bps":    round(-impact, 2),
            "borrow_bps":    round(-borrow, 2),
            "rebalance_bps": round(-rebal,  2),
            "net_bps":       round(net,    2),
        })
    return pd.DataFrame(rows).set_index("regime")
