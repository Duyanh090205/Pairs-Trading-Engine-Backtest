"""
Inspect fold 26 (2025-02) — verify the +207% return is real or a calc artifact.

Re-runs fold 26 with the production config (entry_z=2.0, hl_max=30, dynamic cost)
and reports per-pair breakdown, top-N pairs by P&L, and spread-move sanity check.
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))
sys.path.insert(0, str(WEEK6 / "scripts"))

from run_v4_pipeline import _load_all_daily, _slice_daily, DATA_DAILY, TOTAL_CAPITAL
from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily
from engine_daily.cost_engine import load_cost_data
from engine.phase1_cointegration.factor_residual import project_residual

ENTRY_Z = 2.0
HL_MAX  = 30.0
HARD_SL = 4.0
Z_WIN   = 60


def main() -> int:
    print("=== F26 inspection: 2025-02 (formation 2024-02 to 2025-01) ===\n")

    t0 = time.time()
    cache = _load_all_daily(DATA_DAILY)
    cost_data = load_cost_data()
    print(f"loaded cache + cost data in {time.time()-t0:.1f}s\n")

    # Slice F26 windows
    formation = _slice_daily(cache, "2024-02-01", "2025-01-31")
    trading = _slice_daily(cache, "2025-02-01", "2025-02-28")
    print(f"formation: {len(formation)} tickers, trading: {len(trading)} tickers\n")

    # Discovery
    pairs_df, factor_state = discovery_daily.run(formation, hl_min=5.0, hl_max=HL_MAX)
    print(f"discovery: {len(pairs_df)} pairs after filters\n")

    # Build trading residuals + Path A
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
    print("Running engine with dynamic cost...")
    pair_results = run_fold_daily(
        pairs_df=pairs_df, resid_form=resid_form, resid_trade=resid_trade,
        alpha_lookback=60, entry_z=ENTRY_Z, z_window=Z_WIN, hard_sl_z=HARD_SL,
        cost_data=cost_data,
    )
    print(f"engine ran {len(pair_results)} pairs\n")

    # --- Build per-pair summary ---
    rows = []
    for (ta, tb), df in pair_results.items():
        pos = df["position"].values
        notional = float(df.attrs.get("notional", 0.0))
        pos_prev = np.zeros_like(pos)
        pos_prev[1:] = pos[:-1]
        n_entries = int(((pos != 0) & (pos_prev == 0)).sum())
        gross = float(df["daily_pnl_gross"].sum())
        net = float(df["daily_pnl_net"].sum())
        cost_total = float((df["cost_entry"] + df["cost_exit"] + df["borrow_cost"]).sum())
        # Spread move during trade life
        spread_in_pos = df["spread"][df["position"] != 0]
        if len(spread_in_pos) > 1:
            spread_range = float(spread_in_pos.max() - spread_in_pos.min())
            spread_first = float(spread_in_pos.iloc[0])
            spread_last = float(spread_in_pos.iloc[-1])
        else:
            spread_range = spread_first = spread_last = 0.0
        rows.append({
            "pair": f"{ta}/{tb}",
            "notional": notional,
            "n_trades": n_entries,
            "gross_pnl": gross,
            "net_pnl": net,
            "cost_total": cost_total,
            "pnl_as_pct_notional": 100 * net / max(notional, 1),
            "spread_range_in_pos": spread_range,
        })

    df_summary = pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)

    print("="*100)
    print(f"TOP 10 PAIRS by net P&L ({len(df_summary)} pairs total)")
    print("="*100)
    print(df_summary.head(10).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print()
    print("="*100)
    print("BOTTOM 5 PAIRS by net P&L")
    print("="*100)
    print(df_summary.tail(5).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # --- Concentration analysis ---
    total_net = df_summary["net_pnl"].sum()
    df_summary["pct_of_fold_pnl"] = 100 * df_summary["net_pnl"] / total_net
    df_summary["cum_pct"] = df_summary["pct_of_fold_pnl"].cumsum()

    print()
    print("="*100)
    print("CONCENTRATION: how many pairs drive the fold P&L?")
    print("="*100)
    print(f"Fold total net P&L: ${total_net:,.2f}  (=  {100*total_net/TOTAL_CAPITAL:.2f}% of $1M capital)")
    print(f"Top  1 pair    contributes: {df_summary.iloc[0]['pct_of_fold_pnl']:>6.1f}%  (cum {df_summary.iloc[0]['cum_pct']:>6.1f}%)")
    print(f"Top  3 pairs   contribute:  {df_summary.head(3)['pct_of_fold_pnl'].sum():>6.1f}%  (cum {df_summary.iloc[2]['cum_pct']:>6.1f}%)")
    print(f"Top  5 pairs   contribute:  {df_summary.head(5)['pct_of_fold_pnl'].sum():>6.1f}%  (cum {df_summary.iloc[4]['cum_pct']:>6.1f}%)")
    print(f"Top 10 pairs   contribute:  {df_summary.head(10)['pct_of_fold_pnl'].sum():>6.1f}%  (cum {df_summary.iloc[9]['cum_pct']:>6.1f}%)")
    print(f"Top 20 pairs   contribute:  {df_summary.head(20)['pct_of_fold_pnl'].sum():>6.1f}%  (cum {df_summary.iloc[19]['cum_pct']:>6.1f}%)")

    # --- Suspect: pairs whose P&L > 100% of notional ---
    suspect = df_summary[df_summary["pnl_as_pct_notional"].abs() > 100]
    print()
    print("="*100)
    print(f"PAIRS WITH |P&L| > 100% of notional (red flag for calc issues or huge spread moves)")
    print("="*100)
    print(f"  Found {len(suspect)} suspect pair(s)")
    if len(suspect) > 0:
        print(suspect.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # --- Deep-dive on the #1 pair ---
    print()
    print("="*100)
    print(f"DEEP DIVE: pair #1 by net P&L = {df_summary.iloc[0]['pair']}")
    print("="*100)
    top_pair_name = df_summary.iloc[0]["pair"]
    ta, tb = top_pair_name.split("/")
    df1 = pair_results[(ta, tb)]
    print(f"\n  notional      : ${df1.attrs.get('notional', 0):,.0f}")
    print(f"  n_trades      : {int(((df1['position'].values != 0) & (np.r_[0, df1['position'].values[:-1]] == 0)).sum())}")
    print(f"  net_pnl       : ${df1['daily_pnl_net'].sum():+,.2f}")
    print(f"  gross_pnl     : ${df1['daily_pnl_gross'].sum():+,.2f}")
    print(f"  cost (e+x+b)  : ${(df1['cost_entry']+df1['cost_exit']+df1['borrow_cost']).sum():+,.2f}")
    print(f"\n  Per-day trace:")
    show_cols = ["spread", "zscore", "position", "daily_pnl_gross", "daily_pnl_net", "cost_entry", "cost_exit", "borrow_cost"]
    print(df1[show_cols].to_string(float_format=lambda x: f"{x:+10.4f}"))

    # Save full per-pair summary
    out_path = WEEK6 / "results" / "v4" / "f26_inspection.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
