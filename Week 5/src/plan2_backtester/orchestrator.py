"""
Plan 2 orchestrator: produces data/cost_log.parquet and data/kappa_per_fold.parquet
from Week 4's trade_log + rebalance_log + Plan 0's microstructure parquets.
"""
import pandas as pd
from pathlib import Path

from src.plan1_cost_model.interface_contract import (
    validate_trade_log,
    validate_rebalance_log,
    validate_cost_log,
)
from src.plan2_backtester.walk_forward import FOLD_SCHEDULE, calibrate_fold
from src.plan2_backtester.hooks import (
    trade_dynamic_cost,
    trade_static_cost,
    rebalance_dynamic_cost,
)


WEEK5_ROOT = Path(__file__).resolve().parents[2]
# Week 4 inputs now copied into Week 5/data/week4_inputs/ for self-containment
WEEK4_INPUTS = WEEK5_ROOT / "data" / "week4_inputs"
MICRO_DIR = WEEK5_ROOT / "data" / "microstructure"


def _parse_ts_column(series: pd.Series) -> pd.Series:
    """
    Parse a timestamp column from Week 4 CSVs (mixed-offset Eastern strings)
    into tz-aware US/Eastern datetimes.
    """
    parsed = pd.to_datetime(series, utc=True)
    return parsed.dt.tz_convert("US/Eastern")


def load_trade_log(path: Path | str = None) -> pd.DataFrame:
    """Load Week 4 trade_log.csv and parse timestamps to US/Eastern."""
    path = Path(path) if path else WEEK4_INPUTS / "trade_log.csv"
    df = pd.read_csv(path)
    df["entry_ts"] = _parse_ts_column(df["entry_ts"])
    df["exit_ts"] = _parse_ts_column(df["exit_ts"])
    validate_trade_log(df)
    return df


def load_rebalance_log(path: Path | str = None) -> pd.DataFrame:
    """Load Week 4 rebalance_log.csv and parse timestamps to US/Eastern."""
    path = Path(path) if path else WEEK4_INPUTS / "rebalance_log.csv"
    df = pd.read_csv(path)
    df["rebalance_ts"] = _parse_ts_column(df["rebalance_ts"])
    validate_rebalance_log(df)
    return df


def load_spread_lookup(
    micro_dir: Path | str = None,
    tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads spreads_1min.parquet filtered to `tickers` (traded tickers only).
    spread_rolling columns are already embedded in spreads_1min from the real-data
    runner, so the separate spread_rolling.parquet merge is skipped when
    spread_std_1d is already present.

    Returns (spreads_indexed, spreads_flat):
      spreads_indexed: MultiIndexed [ticker, timestamp_et] for hooks._lookup_at
      spreads_flat:    flat DataFrame for walk_forward.calibrate_fold
    """
    micro_dir = Path(micro_dir) if micro_dir else MICRO_DIR
    parquet_path = micro_dir / "spreads_1min.parquet"

    # Filter to traded tickers at read time — avoids loading all 195M rows
    filters = [("ticker", "in", tickers)] if tickers else None
    spreads = pd.read_parquet(parquet_path, filters=filters)

    # spread_rolling merge: only needed if rolling columns are absent
    if "spread_std_1d" not in spreads.columns:
        rolling_path = micro_dir / "spread_rolling.parquet"
        if rolling_path.exists():
            rolling = pd.read_parquet(rolling_path, filters=filters)
            spreads = spreads.merge(rolling, on=["timestamp_et", "ticker"], how="left")
        else:
            spreads["spread_std_1d"] = float("nan")

    indexed = spreads.set_index(["ticker", "timestamp_et"]).sort_index()
    return indexed, spreads


def run_plan2(
    trade_log_path: Path | str = None,
    rebalance_log_path: Path | str = None,
    micro_dir: Path | str = None,
    out_dir: Path | str = None,
    borrow_rate_bps_annual: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point. Returns (cost_log, kappa_audit) and writes both as parquet.
    """
    out_dir = Path(out_dir) if out_dir else WEEK5_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_log = load_trade_log(trade_log_path)
    rebalance_log = load_rebalance_log(rebalance_log_path)

    # Collect all tickers that appear in the trade log to filter microstructure load
    traded_tickers = sorted(set(
        trade_log["ticker_A"].tolist() + trade_log["ticker_B"].tolist()
    ))
    spreads_idx, spreads_flat = load_spread_lookup(micro_dir, tickers=traded_tickers)

    cost_rows: list[dict] = []
    kappa_rows: list[dict] = []

    for fold in FOLD_SCHEDULE:
        fold_trades = trade_log[trade_log["fold_id"] == fold.fold_id]
        if fold_trades.empty:
            continue

        tickers = (
            pd.concat([fold_trades["ticker_A"], fold_trades["ticker_B"]])
            .unique()
            .tolist()
        )

        kappa_map = calibrate_fold(
            fold.fold_id,
            fold.formation_start,
            fold.formation_end,
            fold.trading_start,
            tickers,
            spreads_flat,
        )
        for t, k in kappa_map.items():
            kappa_rows.append({"fold_id": fold.fold_id, "ticker": t, "kappa": k})

        fold_rebals = rebalance_log[rebalance_log["fold_id"] == fold.fold_id]

        for trade in fold_trades.itertuples():
            d = trade_dynamic_cost(trade, spreads_idx, kappa_map, borrow_rate_bps_annual)
            stc = trade_static_cost(trade)

            trade_rebals = fold_rebals[fold_rebals["trade_id"] == trade.trade_id]
            reb_dollars = sum(
                rebalance_dynamic_cost(r, spreads_idx, kappa_map)
                for r in trade_rebals.itertuples()
            )

            total_dyn = d["total_$"] + reb_dollars
            net_pnl = trade.gross_pnl_dollars - total_dyn

            cost_rows.append({
                "trade_id": trade.trade_id,
                "spread_cost_dollars": d["spread_$"],
                "impact_cost_dollars": d["impact_$"],
                "borrow_cost_dollars": d["borrow_$"],
                "rebalance_cost_dollars": reb_dollars,
                "total_cost_dollars": total_dyn,
                "net_pnl_dollars": net_pnl,
                "net_return": net_pnl / trade.allocated_capital if trade.allocated_capital else 0.0,
                # Comparison regimes
                "total_cost_static": stc,
                "total_cost_gross": 0.0,
                "net_pnl_static": trade.gross_pnl_dollars - stc,
                "gross_pnl_dollars": trade.gross_pnl_dollars,
            })

    cost_log = pd.DataFrame(cost_rows)
    kappa_audit = pd.DataFrame(kappa_rows)

    validate_cost_log(cost_log)

    cost_log.to_parquet(out_dir / "cost_log.parquet")
    kappa_audit.to_parquet(out_dir / "kappa_per_fold.parquet")

    return cost_log, kappa_audit


if __name__ == "__main__":
    cl, ka = run_plan2()
    print(f"Wrote cost_log.parquet ({len(cl)} rows)")
    print(f"Wrote kappa_per_fold.parquet ({len(ka)} rows)")
