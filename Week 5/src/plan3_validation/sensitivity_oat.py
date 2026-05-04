"""
4.8 — OAT sensitivity sweep.

9 OAT runs:
  kappa multiplier {0.5, 1.0, 1.5}
  borrow rate {30, 50, 100} bps
  spread level {L1, L2}    [L2 noted as deferred — would require full re-run]

Re-aggregation strategy (no re-loop over trades):
  - kappa multiplier: scales impact_cost_dollars linearly
  - borrow rate ratio: scales borrow_cost_dollars linearly
  - L2: deferred (logged with explicit note)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.plan3_validation.sharpe_net import _load_gross_daily_returns, DEFAULT_EQUITY_DIR


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"


def _net_sharpe_after_perturb(
    cost_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    impact_mult: float = 1.0,
    borrow_mult: float = 1.0,
) -> float:
    gross_daily = _load_gross_daily_returns(DEFAULT_EQUITY_DIR)
    if gross_daily.empty:
        return float("nan")

    aum = trade_log["allocated_capital"].sum()
    if aum <= 0:
        return float("nan")

    df = cost_log.merge(trade_log[["trade_id", "exit_ts"]], on="trade_id")
    df["exit_dt"] = (
        pd.to_datetime(df["exit_ts"], utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.normalize()
    )
    perturbed_cost = (
        df["spread_cost_dollars"]
        + df["impact_cost_dollars"] * impact_mult
        + df["borrow_cost_dollars"] * borrow_mult
        + df["rebalance_cost_dollars"]
    )
    cost_by_day = df.assign(cost=perturbed_cost).groupby("exit_dt")["cost"].sum() / aum
    cost_adj = cost_by_day.reindex(gross_daily.index, fill_value=0.0)

    daily = (gross_daily - cost_adj).to_numpy()
    if len(daily) < 2:
        return float("nan")
    sd = daily.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(daily.mean() / sd * np.sqrt(252))


def run_oat_sensitivity(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
) -> pd.DataFrame:
    """
    Returns 9-row DataFrame:
      kappa_mult, borrow_bps, spread_level, net_sharpe, delta_vs_baseline.
    """
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)

    baseline_sharpe = _net_sharpe_after_perturb(cost_log, trade_log, 1.0, 1.0)

    rows: list[dict] = []
    for k_mult in (0.5, 1.0, 1.5):
        for borrow in (30.0, 50.0, 100.0):
            borrow_mult = borrow / 50.0
            sh = _net_sharpe_after_perturb(cost_log, trade_log, k_mult, borrow_mult)
            rows.append({
                "kappa_mult":        k_mult,
                "borrow_bps":        borrow,
                "spread_level":      "L1",
                "net_sharpe":        round(sh, 4) if not np.isnan(sh) else float("nan"),
                "delta_vs_baseline": round(sh - baseline_sharpe, 4)
                                      if not (np.isnan(sh) or np.isnan(baseline_sharpe))
                                      else float("nan"),
            })

    rows.append({
        "kappa_mult": float("nan"), "borrow_bps": float("nan"),
        "spread_level": "L2",
        "net_sharpe": float("nan"),
        "delta_vs_baseline": float("nan"),
    })
    return pd.DataFrame(rows)
