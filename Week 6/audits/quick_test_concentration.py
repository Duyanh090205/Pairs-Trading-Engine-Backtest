"""
quick_test_concentration.py — Test B4 (concentration)
======================================================
Test top-K pair selection on same 5 representative folds.

Approach: run discovery once per fold (cached), then re-run engine with
different K values (top-10, 20, 50, 100, all). Discovery is the slow part
(~30s), engine is fast (~5s), so this is dominated by 5 discoveries.

Settings:
    K = 10, 20, 50, 100, 500 (all = baseline)

Folds: same 5 as quick_test_options.py (1, 8, 18, 25, 33).

Runtime: ~5 min (5 discoveries + 25 engine runs).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from scripts.run_v4_pipeline import (
    DATA_DAILY, FOLD_SCHEDULE, _load_all_daily, _slice_daily, TOTAL_CAPITAL,
)
from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily
from engine_daily.metrics_daily import aggregate_fold_metrics
from engine_daily.cost_engine import load_cost_data
from engine.phase1_cointegration.factor_residual import project_residual


REPRESENTATIVE_FOLDS = [1, 8, 18, 25, 33]
K_VALUES = [10, 20, 50, 100, 500]


def main() -> int:
    print(f"=== Quick test: top-K concentration ===\n", flush=True)
    print(f"Folds: {REPRESENTATIVE_FOLDS}", flush=True)
    print(f"K values (top-K by Johansen pval): {K_VALUES}\n", flush=True)

    print("Loading cache + cost data...", flush=True)
    t0 = time.time()
    cache_daily = _load_all_daily(DATA_DAILY)
    cost_data = load_cost_data()
    print(f"  Loaded in {time.time()-t0:.1f}s\n", flush=True)

    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}
    results: list[dict] = []

    for fold_n in REPRESENTATIVE_FOLDS:
        sched = sched_by_fold.get(fold_n)
        if sched is None:
            continue
        _, fs, fe, tm = sched
        print(f"--- Fold {fold_n} ({tm}) ---", flush=True)
        t_disc = time.time()

        # ---- Run discovery once (cache result) ----
        formation_daily = _slice_daily(cache_daily, fs, fe)
        if not formation_daily:
            continue
        pairs_df, factor_state = discovery_daily.run(formation_data=formation_daily)
        print(f"  Discovery: {len(pairs_df)} pairs in {time.time()-t_disc:.1f}s",
              flush=True)

        # ---- Compute trading residuals (Path A re-anchor) ----
        trade_start = tm + "-01"
        trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        trading_daily_raw = _slice_daily(cache_daily, trade_start, trade_end)
        if not trading_daily_raw:
            continue

        loadings_W = factor_state["loadings_W"]
        factor_tickers = factor_state["tickers"]
        resid_form_df = factor_state["residual_log_prices"]

        resid_trade_dict = project_residual(
            trading_daily_raw, loadings_W, factor_tickers, min_obs=10,
        )
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form_df.columns:
                continue
            s_form_tk = resid_form_df[tk].dropna()
            if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
                continue
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
        resid_trade_df = pd.concat(resid_trade_dict, axis=1)

        # ---- For each K, restrict pair list and run engine ----
        # Sort pairs by johansen_pval ascending so top-K = strongest signals
        pairs_sorted = pairs_df.sort_values("johansen_pval")
        for K in K_VALUES:
            pairs_topk = pairs_sorted.head(K)
            t_eng = time.time()
            pair_results = run_fold_daily(
                pairs_df=pairs_topk,
                resid_form=resid_form_df,
                resid_trade=resid_trade_df,
                cost_data=cost_data,
                current_fold_n=fold_n,
            )
            fm = aggregate_fold_metrics(pair_results, total_capital=TOTAL_CAPITAL)
            eb = fm.get("exit_breakdown", {})
            elapsed_eng = time.time() - t_eng
            results.append({
                "K": K, "fold": fold_n, "trading_month": tm,
                "n_pairs_in": len(pairs_topk),
                "n_pairs_ran": len(pair_results),
                "n_trades": fm.get("n_trades", 0),
                "sharpe": fm.get("sharpe", 0.0),
                "total_return": fm.get("total_return", 0.0),
                "avg_net_bps": fm.get("avg_net_bps", 0.0),
                "n_zero_cross": eb.get("zero_cross", 0),
                "n_hard_sl": eb.get("hard_sl", 0),
                "n_open_at_eom": eb.get("open_at_eom", 0),
            })
            print(f"  K={K:>4}: pairs_ran={len(pair_results):>3}, trades={fm['n_trades']:>3}, "
                  f"sharpe={fm['sharpe']:+.3f}, ret={fm['total_return']:+.4f}, "
                  f"zc/sl/eom={eb.get('zero_cross',0):>2}/{eb.get('hard_sl',0):>2}/"
                  f"{eb.get('open_at_eom',0):>3} ({elapsed_eng:.1f}s)", flush=True)
        print()

    df = pd.DataFrame(results)
    out_csv = WEEK6 / "results" / "v4" / "quick_test_concentration.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)\n")

    print("=== Summary by K (averaged across 5 representative folds) ===")
    summary = df.groupby("K").agg(
        n_folds=("fold", "count"),
        avg_pairs_ran=("n_pairs_ran", "mean"),
        sum_trades=("n_trades", "sum"),
        mean_sharpe=("sharpe", "mean"),
        median_sharpe=("sharpe", "median"),
        sum_return=("total_return", "sum"),
        avg_net_bps=("avg_net_bps", "mean"),
        sum_zc=("n_zero_cross", "sum"),
        sum_sl=("n_hard_sl", "sum"),
        sum_eom=("n_open_at_eom", "sum"),
    ).round(3)
    print(summary.to_string())

    print()
    print("=== Sharpe by fold and K ===")
    pivot = df.pivot(index="fold", columns="K", values="sharpe")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
