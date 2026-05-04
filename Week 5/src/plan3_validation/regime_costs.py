"""
4.3 — Regime-conditional cost analysis.

Splits trades into 4 regimes per Week 4's REGIME_MAP:
  - Late Bear 2022       (folds 1-6)
  - Early Bull 2023      (folds 7-18)
  - Mid Bull 2024        (folds 19-30)
  - Late Bull 2025-Q12026 (folds 31-45)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"
DEFAULT_SPREADS = WEEK5_ROOT / "data" / "microstructure" / "spreads_1min.parquet"


REGIME_MAP = {
    "Late Bear 2022":         range(1, 7),
    "Early Bull 2023":        range(7, 19),
    "Mid Bull 2024":          range(19, 31),
    "Late Bull 2025-Q12026":  range(31, 46),
}


def _fold_to_regime(fold_id: int) -> str:
    for name, rng in REGIME_MAP.items():
        if fold_id in rng:
            return name
    return "Unknown"


def _annualised_sharpe(daily: np.ndarray) -> float:
    arr = daily[~np.isnan(daily)]
    if len(arr) < 2:
        return float("nan")
    sd = arr.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(252))


def analyze_regime_costs(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
    spreads_path: Path | str = DEFAULT_SPREADS,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by regime name with columns:
      n_trades, avg_l1_spread_bps, avg_dyn_rt_cost_bps,
      sharpe_gross, sharpe_dynamic, delta_sharpe.
    """
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)
    df = cost_log.merge(trade_log[["trade_id", "fold_id", "allocated_capital", "exit_ts"]], on="trade_id")
    df["regime"] = df["fold_id"].apply(_fold_to_regime)
    df["exit_date"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("US/Eastern").dt.date

    # Filter spreads to traded tickers only — avoids loading 195M rows
    all_traded = pd.concat(
        [trade_log["ticker_A"], trade_log["ticker_B"]]
    ).unique().tolist()
    spreads = pd.read_parquet(
        spreads_path,
        columns=["ticker", "full_spread_l1_bps"],
        filters=[("ticker", "in", all_traded)],
    )

    aum = trade_log["allocated_capital"].sum()

    rows: list[dict] = []
    for regime in REGIME_MAP:
        sub = df[df["regime"] == regime]
        if sub.empty:
            rows.append({
                "regime": regime, "n_trades": 0,
                "avg_l1_spread_bps": float("nan"),
                "avg_dyn_rt_cost_bps": float("nan"),
                "sharpe_gross": float("nan"),
                "sharpe_dynamic": float("nan"),
                "delta_sharpe": float("nan"),
            })
            continue

        avg_cost_bps = (sub["total_cost_dollars"] / sub["allocated_capital"]).mean() * 10_000

        # Sharpe gross/dynamic from daily PnL aggregated across this regime's trades.
        # Use explicit subtract-then-groupby to avoid pandas 2.2+ apply deprecation.
        sub_with_net = sub.assign(net_pnl=sub["gross_pnl_dollars"] - sub["total_cost_dollars"])
        d_gross = sub.groupby("exit_date")["gross_pnl_dollars"].sum() / aum
        d_dyn = sub_with_net.groupby("exit_date")["net_pnl"].sum() / aum

        # C2: zero-fill within this regime's exit-date span so std() is not
        # computed over only the ~10-50 exit dates in each sub-period.
        if len(d_gross) >= 2:
            regime_idx = pd.bdate_range(
                start=pd.Timestamp(d_gross.index.min()),
                end=pd.Timestamp(d_gross.index.max()),
            )
            d_gross = d_gross.reindex(regime_idx, fill_value=0.0)
            d_dyn = d_dyn.reindex(regime_idx, fill_value=0.0)

        sharpe_gross = _annualised_sharpe(d_gross.to_numpy())
        sharpe_dyn = _annualised_sharpe(d_dyn.to_numpy())

        # Avg L1 spread: filter spreads to (a) tickers in this regime's trades
        # AND (b) timestamps inside fold range
        regime_tickers = pd.concat([
            trade_log.loc[trade_log["fold_id"].isin(REGIME_MAP[regime]), "ticker_A"],
            trade_log.loc[trade_log["fold_id"].isin(REGIME_MAP[regime]), "ticker_B"],
        ]).unique()
        sp_slice = spreads[spreads["ticker"].isin(regime_tickers)]
        avg_l1 = sp_slice["full_spread_l1_bps"].mean() if len(sp_slice) else float("nan")

        rows.append({
            "regime": regime,
            "n_trades": len(sub),
            "avg_l1_spread_bps":   round(avg_l1, 2) if not np.isnan(avg_l1) else float("nan"),
            "avg_dyn_rt_cost_bps": round(avg_cost_bps, 2),
            "sharpe_gross":        round(sharpe_gross, 3) if not np.isnan(sharpe_gross) else float("nan"),
            "sharpe_dynamic":      round(sharpe_dyn, 3) if not np.isnan(sharpe_dyn) else float("nan"),
            "delta_sharpe":        round(sharpe_dyn - sharpe_gross, 3)
                                    if not (np.isnan(sharpe_dyn) or np.isnan(sharpe_gross))
                                    else float("nan"),
        })

    return pd.DataFrame(rows).set_index("regime")
