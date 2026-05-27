"""
Audit: do high-β pairs systematically WIN or LOSE in the V4 dynamic-cost run?

For each of the 39 folds:
  1. Run discovery → capture pairs_df (β per pair)
  2. Run engine (dynamic cost) → capture pair_results (PnL per pair)
  3. Save per-pair row: (fold, ticker_a, ticker_b, beta_pca, notional, n_trades,
     gross_pnl, net_pnl, pct_of_fold_pnl)

Then analyze:
  - β bucket → mean PnL, win rate, contribution to fold P&L
  - Question: are extreme-β pairs net-positive (real alpha), net-negative
    (predominantly losers), or just high-variance (50/50 with extreme outcomes)?
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))
sys.path.insert(0, str(WEEK6 / "scripts"))
logging.basicConfig(level=logging.WARNING)

from run_v4_pipeline import _load_all_daily, _slice_daily, FOLD_SCHEDULE, DATA_DAILY
from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily
from engine_daily.cost_engine import load_cost_data
from engine.phase1_cointegration.factor_residual import project_residual

ENTRY_Z = 2.0
HL_MAX  = 30.0
HARD_SL = 4.0
Z_WIN   = 60


def main() -> int:
    print("=== V4 β vs P&L audit (39 folds, dynamic cost) ===\n")
    t0 = time.time()
    cache = _load_all_daily(DATA_DAILY)
    cost_data = load_cost_data()
    print(f"loaded cache + cost data in {time.time()-t0:.1f}s\n")

    rows = []
    for fold_n, fs, fe, tm in FOLD_SCHEDULE:
        t_fold = time.time()
        formation = _slice_daily(cache, fs, fe)
        if not formation:
            continue
        try:
            pairs_df, factor_state = discovery_daily.run(formation, hl_min=5.0, hl_max=HL_MAX)
        except Exception as e:
            print(f"  Fold {fold_n}: discovery failed: {e}")
            continue
        if pairs_df.empty:
            continue

        # Trading residuals + Path A
        trade_start = tm + "-01"
        trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        trading = _slice_daily(cache, trade_start, trade_end)
        if not trading:
            continue
        W = factor_state["loadings_W"]
        fact_tk = factor_state["tickers"]
        resid_form = factor_state["residual_log_prices"]
        resid_trade_dict = project_residual(trading, W, fact_tk, min_obs=10)
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form.columns:
                continue
            s_form_tk = resid_form[tk].dropna()
            if len(s_form_tk) and len(resid_trade_dict[tk]):
                shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
                resid_trade_dict[tk] = resid_trade_dict[tk] + shift
        resid_trade = pd.concat(resid_trade_dict, axis=1)

        # Engine with dynamic cost
        pair_results = run_fold_daily(
            pairs_df=pairs_df, resid_form=resid_form, resid_trade=resid_trade,
            alpha_lookback=60, entry_z=ENTRY_Z, z_window=Z_WIN, hard_sl_z=HARD_SL,
            cost_data=cost_data,
        )

        # Join β from pairs_df with PnL from pair_results
        beta_lookup = {(r["ticker_a"], r["ticker_b"]): float(r["beta_pca"])
                       for _, r in pairs_df.iterrows()}

        fold_total_pnl = sum(df["daily_pnl_net"].sum() for df in pair_results.values())
        for (ta, tb), df in pair_results.items():
            beta = beta_lookup.get((ta, tb), np.nan)
            pos = df["position"].values
            pos_prev = np.zeros_like(pos)
            pos_prev[1:] = pos[:-1]
            n_entries = int(((pos != 0) & (pos_prev == 0)).sum())
            gross = float(df["daily_pnl_gross"].sum())
            net = float(df["daily_pnl_net"].sum())
            notional = float(df.attrs.get("notional", 0.0))
            rows.append({
                "fold": fold_n,
                "trading_month": tm,
                "ticker_a": ta, "ticker_b": tb,
                "beta_pca": beta,
                "abs_beta": abs(beta),
                "notional": notional,
                "n_trades": n_entries,
                "gross_pnl": gross,
                "net_pnl": net,
                "pct_of_fold_pnl": 100.0 * net / fold_total_pnl if abs(fold_total_pnl) > 1 else np.nan,
                "fold_total_pnl": fold_total_pnl,
            })

        elapsed = time.time() - t_fold
        max_beta = max(abs(beta_lookup.get((ta, tb), 0)) for (ta, tb) in pair_results.keys())
        print(f"  Fold {fold_n:02d} [{tm}]: pairs={len(pair_results):>3}  "
              f"max|β|={max_beta:>8.2f}  fold_pnl=${fold_total_pnl:>+11,.0f}  "
              f"({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    out_path = WEEK6 / "results" / "v4" / "beta_vs_pnl_all_folds.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df):,} pair-fold rows)\n")

    # =================================================
    # ANALYSIS
    # =================================================
    print("="*80)
    print("DO HIGH-β PAIRS MAKE OR LOSE MONEY?")
    print("="*80)
    bins = [0, 1, 2, 3, 5, 10, 50, 1e9]
    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,5)", "[5,10)", "[10,50)", "[50,∞)"]
    df["beta_bin"] = pd.cut(df["abs_beta"], bins=bins, labels=labels, right=False)

    agg = df.groupby("beta_bin", observed=True).agg(
        n_pairs=("net_pnl", "count"),
        sum_net=("net_pnl", "sum"),
        mean_net=("net_pnl", "mean"),
        median_net=("net_pnl", "median"),
        std_net=("net_pnl", "std"),
        pct_winning=("net_pnl", lambda x: 100 * (x > 0).mean()),
        max_abs_net=("net_pnl", lambda x: x.abs().max()),
    ).round(2)
    print(agg.to_string())

    print()
    print("="*80)
    print("CONTRIBUTION TO TOTAL P&L by β bucket")
    print("="*80)
    total_run_pnl = df["net_pnl"].sum()
    contrib = df.groupby("beta_bin", observed=True).agg(
        n_pairs=("net_pnl", "count"),
        sum_net=("net_pnl", "sum"),
    )
    contrib["pct_of_total_pnl"] = 100 * contrib["sum_net"] / total_run_pnl
    contrib["pct_of_pairs"] = 100 * contrib["n_pairs"] / len(df)
    print(contrib.round(2).to_string())
    print(f"\n  Total run P&L: ${total_run_pnl:,.0f}")

    print()
    print("="*80)
    print("TOP-10 MOST EXTREME |β| PAIRS — did they win or lose?")
    print("="*80)
    top_ext = df.nlargest(10, "abs_beta")[
        ["fold", "trading_month", "ticker_a", "ticker_b", "beta_pca",
         "n_trades", "gross_pnl", "net_pnl", "pct_of_fold_pnl"]
    ]
    print(top_ext.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print()
    print("="*80)
    print("VARIANCE EFFECT — std of P&L by β bucket (are high-β pairs just noisier?)")
    print("="*80)
    print("  (Higher std relative to mean = noisier, not necessarily losing)")
    summary = df.groupby("beta_bin", observed=True).agg(
        n=("net_pnl", "count"),
        mean=("net_pnl", "mean"),
        std=("net_pnl", "std"),
    ).round(2)
    summary["sharpe_like"] = summary["mean"] / summary["std"].replace(0, np.nan)
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
