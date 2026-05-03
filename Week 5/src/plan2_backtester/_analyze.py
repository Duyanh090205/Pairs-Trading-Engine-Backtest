"""Analysis script for Plan 2 outputs. One-shot, not part of the pipeline."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

tl = pd.read_csv("data/week4_inputs/trade_log.csv")
rb = pd.read_csv("data/week4_inputs/rebalance_log.csv")
fm = pd.read_csv("data/week4_inputs/fold_metrics.csv")
cl = pd.read_parquet("data/cost_log.parquet")
ka = pd.read_parquet("data/kappa_per_fold.parquet")

print("=" * 70)
print("1. INPUT INVENTORY")
print("=" * 70)
print(f"trade_log:     {len(tl)} trades across {tl['fold_id'].nunique()} folds")
print(f"rebalance_log: {len(rb)} rebalances across {rb['fold_id'].nunique()} folds")
print(f"fold_metrics:  {len(fm)} fold rows (Week 4 'final' run)")

print("\n" + "=" * 70)
print("2. WEEK 4 BASELINE (uses Week 4 static-30bps-per-leg cost)")
print("=" * 70)
print(f"Folds traded:        {(fm['n_trades'] > 0).sum()} / {len(fm)}")
print(f"Folds with sharpe>0: {(fm['sharpe'] > 0).sum()} / {(fm['n_trades'] > 0).sum()}")
print(f"Median fold Sharpe:  {fm.loc[fm['n_trades']>0, 'sharpe'].median():.3f}")
print(f"Mean cost_commission per fold: ${fm['cost_commission'].mean():.2f}")
print(f"Total cost_commission:         ${fm['cost_commission'].sum():.2f}")
print(f"Total cost_borrow:             ${fm['cost_borrow'].sum():.2f}")
print(f"Total cost_rebalance:          ${fm['cost_rebalance'].sum():.2f}")

print("\n" + "=" * 70)
print("3. PLAN 2 COST_LOG  ---  DESCRIPTIVE STATS (per trade, $)")
print("=" * 70)
desc_cols = ["spread_cost_dollars", "impact_cost_dollars", "borrow_cost_dollars",
             "rebalance_cost_dollars", "total_cost_dollars", "total_cost_static",
             "gross_pnl_dollars", "net_pnl_dollars", "net_pnl_static"]
print(cl[desc_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())

print("\n" + "=" * 70)
print("4. THREE REGIMES SIDE-BY-SIDE (totals across all 90 trades)")
print("=" * 70)
gross_pnl = cl["gross_pnl_dollars"].sum()
static_cost = cl["total_cost_static"].sum()
dyn_cost = cl["total_cost_dollars"].sum()
print(f"  Gross PnL:                   ${gross_pnl:>12,.2f}")
print(f"  Static-60bps total cost:     ${static_cost:>12,.2f}")
print(f"  Dynamic total cost:          ${dyn_cost:>12,.2f}")
print(f"  --- Net PnL by regime ---")
print(f"  Gross net (= gross PnL):     ${gross_pnl:>12,.2f}")
print(f"  Static net:                  ${(gross_pnl - static_cost):>12,.2f}")
print(f"  Dynamic net:                 ${(gross_pnl - dyn_cost):>12,.2f}")
if gross_pnl != 0:
    print(f"  --- Cost as % of |gross PnL| ---")
    print(f"  Static:  {(static_cost / abs(gross_pnl)) * 100:>6.1f}%")
    print(f"  Dynamic: {(dyn_cost / abs(gross_pnl)) * 100:>6.1f}%")

print("\n" + "=" * 70)
print("5. COMPONENT WATERFALL (mean per trade, $)")
print("=" * 70)
mean_total = cl["total_cost_dollars"].mean()
print(f"  Spread cost:    ${cl['spread_cost_dollars'].mean():>8.2f}  ({cl['spread_cost_dollars'].mean()/mean_total*100:5.1f}% of dynamic)")
print(f"  Impact cost:    ${cl['impact_cost_dollars'].mean():>8.2f}  ({cl['impact_cost_dollars'].mean()/mean_total*100:5.1f}%)")
print(f"  Borrow cost:    ${cl['borrow_cost_dollars'].mean():>8.2f}  ({cl['borrow_cost_dollars'].mean()/mean_total*100:5.1f}%)")
print(f"  Rebalance cost: ${cl['rebalance_cost_dollars'].mean():>8.2f}  ({cl['rebalance_cost_dollars'].mean()/mean_total*100:5.1f}%)")
print(f"  ---")
print(f"  Total dynamic:  ${mean_total:>8.2f}")

print("\n" + "=" * 70)
print("6. PROFITABILITY UNDER EACH REGIME")
print("=" * 70)
n = len(cl)
print(f"  Trades profitable Gross:   {(cl['gross_pnl_dollars']>0).sum():3d} / {n}  ({(cl['gross_pnl_dollars']>0).mean()*100:.1f}%)")
print(f"  Trades profitable Static:  {(cl['net_pnl_static']>0).sum():3d} / {n}  ({(cl['net_pnl_static']>0).mean()*100:.1f}%)")
print(f"  Trades profitable Dynamic: {(cl['net_pnl_dollars']>0).sum():3d} / {n}  ({(cl['net_pnl_dollars']>0).mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("7. KAPPA TIER DISTRIBUTION")
print("=" * 70)
print(ka["kappa"].value_counts().sort_index().to_string())
n_unique_tickers = ka["ticker"].nunique()
ticker_kappa_counts = ka.groupby("ticker")["kappa"].nunique()
print(f"\nTotal (fold,ticker) pairs: {len(ka)}")
print(f"Unique tickers:            {n_unique_tickers}")
print(f"Tickers stable across folds: {(ticker_kappa_counts == 1).sum()} / {n_unique_tickers}")
print(f"Tickers with kappa drift:    {(ticker_kappa_counts > 1).sum()} / {n_unique_tickers}")

print("\n" + "=" * 70)
print("8. RED FLAG TRIGGERS")
print("=" * 70)
from src.plan3_validation.red_flags import (
    check_dynamic_cost_blowup, check_math_violation,
)
mean_rt_bps = (cl["total_cost_dollars"] / 20000).mean() * 10000
print(f"  Mean RT cost (vs $20k allocated): {mean_rt_bps:.1f} bps")
print(f"    -> dynamic_cost_blowup (>150bps trigger): {check_dynamic_cost_blowup(mean_rt_bps)}")
math_viol = ((cl["total_cost_dollars"] < 0) | (cl["net_pnl_dollars"] > cl["gross_pnl_dollars"])).any()
print(f"  Math violation: {math_viol}")
ticker_kappa_changes = (ka.groupby("ticker")["kappa"].nunique() > 1).sum()
print(f"  Tickers with kappa drift across folds: {ticker_kappa_changes} (instability flag if > 2x)")

print("\n" + "=" * 70)
print("9. PER-FOLD COST SUMMARY (first 12 trading folds)")
print("=" * 70)
fold_summary = cl.merge(tl[["trade_id", "fold_id", "allocated_capital"]], on="trade_id")
fs = fold_summary.groupby("fold_id").agg(
    n=("trade_id", "count"),
    gross=("gross_pnl_dollars", "sum"),
    dyn=("total_cost_dollars", "sum"),
    static=("total_cost_static", "sum"),
    rebal=("rebalance_cost_dollars", "sum"),
).round(2)
fs["net_dyn"] = (fs["gross"] - fs["dyn"]).round(2)
fs["net_stat"] = (fs["gross"] - fs["static"]).round(2)
print(fs.head(12).to_string())

print("\n" + "=" * 70)
print("10. INTRADAY VS OVERNIGHT TRADES")
print("=" * 70)
tl["entry_ts"] = pd.to_datetime(tl["entry_ts"], utc=True).dt.tz_convert("US/Eastern")
tl["exit_ts"] = pd.to_datetime(tl["exit_ts"], utc=True).dt.tz_convert("US/Eastern")
intraday_mask = tl["entry_ts"].dt.date == tl["exit_ts"].dt.date
intraday_ids = set(tl.loc[intraday_mask, "trade_id"])
intra = cl[cl["trade_id"].isin(intraday_ids)]
overn = cl[~cl["trade_id"].isin(intraday_ids)]
print(f"  Intraday: n={len(intra):2d}, mean total ${intra['total_cost_dollars'].mean():>6.2f}, mean borrow ${intra['borrow_cost_dollars'].mean():.2f}")
print(f"  Overnight n={len(overn):2d}, mean total ${overn['total_cost_dollars'].mean():>6.2f}, mean borrow ${overn['borrow_cost_dollars'].mean():.2f}")

print("\n" + "=" * 70)
print("11. HOLDING PERIOD VS COST")
print("=" * 70)
tl["holding_days"] = (tl["exit_ts"] - tl["entry_ts"]).dt.total_seconds() / 86400.0
mer = cl.merge(tl[["trade_id", "holding_days", "n_bars"]], on="trade_id")
print(f"  Mean holding days:   {mer['holding_days'].mean():.2f}")
print(f"  Mean bars per trade: {mer['n_bars'].mean():.1f}")
print(f"  corr(holding_days, total_cost):  {mer['holding_days'].corr(mer['total_cost_dollars']):>6.3f}")
print(f"  corr(holding_days, borrow_cost): {mer['holding_days'].corr(mer['borrow_cost_dollars']):>6.3f}")
print(f"  corr(n_bars,       rebal_cost):  {mer['n_bars'].corr(mer['rebalance_cost_dollars']):>6.3f}")
