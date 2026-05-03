"""
4.5 — Kill zone + intraday spread seasonality.

Two outputs from one computation:
  Part A: avg spread by 30-min bucket x liquidity tier (U-shape validation).
  Part B: avg net alpha by 30-min bucket (kill zone heatmap).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"
DEFAULT_SPREADS = WEEK5_ROOT / "data" / "microstructure" / "spreads_1min.parquet"
DEFAULT_KAPPA = WEEK5_ROOT / "data" / "kappa_per_fold.parquet"


_BUCKET_LABELS = [
    "09:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-11:30",
    "11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30",
    "13:30-14:00", "14:00-14:30", "14:30-15:00", "15:00-15:30", "15:30-15:59",
]


def _bucket_for_ts(ts: pd.Timestamp) -> str:
    h, m = ts.hour, ts.minute
    minutes = h * 60 + m
    start = 9 * 60 + 30
    idx = (minutes - start) // 30
    if idx < 0:
        return _BUCKET_LABELS[0]
    if idx >= len(_BUCKET_LABELS):
        return _BUCKET_LABELS[-1]
    return _BUCKET_LABELS[int(idx)]


def analyze_kill_zone(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
    spreads_path: Path | str = DEFAULT_SPREADS,
    kappa_path: Path | str = DEFAULT_KAPPA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (seasonality_table, kill_zone_table).

    seasonality_table: index=bucket, columns=Tier1/Tier2/Tier3, values=avg_spread_bps.
    kill_zone_table: index=bucket, columns=[n_trades, gross_bps, cost_bps, net_bps, kill_zone].
    """
    kappa = pd.read_parquet(kappa_path)
    # Filter spreads to the ~108 traded tickers only — avoids loading 195M rows
    traded_tickers = kappa["ticker"].unique().tolist()
    spreads = pd.read_parquet(
        spreads_path,
        columns=["timestamp_et", "ticker", "full_spread_l1_bps"],
        filters=[("ticker", "in", traded_tickers)],
    )

    # ---------- Part A: U-shape ----------
    # Vectorised bucket assignment (avoids 40M-row Python apply)
    ts = pd.to_datetime(spreads["timestamp_et"])
    minutes = ts.dt.hour * 60 + ts.dt.minute
    idx = ((minutes - (9 * 60 + 30)) // 30).clip(0, len(_BUCKET_LABELS) - 1).to_numpy()
    labels = np.array(_BUCKET_LABELS)
    spreads["bucket"] = labels[idx]

    # Tier per ticker = the most common kappa across that ticker's folds
    ticker_kappa = kappa.groupby("ticker")["kappa"].agg(lambda s: s.mode().iloc[0])
    spreads["tier"] = spreads["ticker"].map(ticker_kappa).map(
        {0.3: "Tier1_Tight", 0.5: "Tier2_Medium", 0.8: "Tier3_Wide"}
    )

    seasonality = (
        spreads.dropna(subset=["tier"])
        .groupby(["bucket", "tier"])["full_spread_l1_bps"]
        .mean()
        .unstack("tier")
        .reindex(_BUCKET_LABELS)
        .round(2)
    )

    # ---------- Part B: kill zones from trades ----------
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)
    df = cost_log.merge(trade_log[["trade_id", "entry_ts", "allocated_capital"]], on="trade_id")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("US/Eastern")
    df["bucket"] = df["entry_ts"].apply(_bucket_for_ts)
    df["gross_bps"] = (df["gross_pnl_dollars"] / df["allocated_capital"]) * 10_000
    df["cost_bps"] = (df["total_cost_dollars"] / df["allocated_capital"]) * 10_000
    df["net_bps"] = df["gross_bps"] - df["cost_bps"]

    kill = (
        df.groupby("bucket")
        .agg(n_trades=("trade_id", "count"),
             gross_bps=("gross_bps", "mean"),
             cost_bps=("cost_bps", "mean"),
             net_bps=("net_bps", "mean"))
        .reindex(_BUCKET_LABELS)
        .round(2)
    )
    kill["kill_zone"] = kill["net_bps"] < 0

    return seasonality, kill
