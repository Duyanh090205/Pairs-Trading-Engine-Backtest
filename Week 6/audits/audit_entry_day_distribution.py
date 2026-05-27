"""
audit_entry_day_distribution.py — Confirm cause B
==================================================
Hypothesis: a large fraction of V4 trades enter too late in the trading
month to mean-revert before EOM force-close, regardless of cointegration
quality. This is a structural mismatch between HL filter [5, 30] days
and the ~21-trading-day window.

Methodology:
    1. Run a single fold via existing run_fold_v4 (no engine modification).
    2. Walk per-pair bar-level DataFrames, extract per-trade:
         entry_day_in_window, exit_day_in_window, duration, exit_code, net_pnl
       using the SAME pattern as metrics_daily._per_pair_stats (position
       transitions + lagged exit_code lookup -- avoids the bug-2 double-count).
    3. Bucket by entry-day {Early / Mid / Late}, report exit-type mix and P&L.

Run:
    python audits/audit_entry_day_distribution.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from scripts.run_v4_pipeline import (
    DATA_DAILY,
    FOLD_SCHEDULE,
    _load_all_daily,
    run_fold_v4,
)
from engine_daily.cost_engine import load_cost_data


# Pick a fold with many trades for good statistics.
# Capped run (β≤5): fold 25 (2025-01) had 112 trades; fold 22 (2024-10) had 110.
FOLD_N = 25


def extract_trades(pair_df: pd.DataFrame) -> list[dict]:
    """Walk one pair's bar-level DataFrame, emit one record per trade.

    Pattern matches metrics_daily._per_pair_stats so counts reconcile.
    """
    pos = pair_df["position"].values
    pnl_net = pair_df["daily_pnl_net"].values
    exit_codes = pair_df["exit_code"].values
    index = pair_df.index
    n_bars = len(pos)

    pos_prev = np.zeros_like(pos)
    pos_prev[1:] = pos[:-1]
    is_entry = (pos != 0) & (pos_prev == 0)
    is_exit_close = (pos == 0) & (pos_prev != 0)

    trades: list[dict] = []
    cur_entry = -1
    for i in range(n_bars):
        if is_entry[i]:
            cur_entry = i
        elif is_exit_close[i] and cur_entry >= 0:
            code = int(exit_codes[i - 1]) if i >= 1 else 0
            trades.append({
                "entry_idx": cur_entry,
                "exit_idx": i,
                "duration_bars": i - cur_entry,
                "exit_code": code,                  # 1=zero, 2=hard SL
                "exit_type": "zero_cross" if code == 1 else
                             "hard_sl" if code == 2 else "unknown",
                "net_pnl": float(pnl_net[cur_entry:i + 1].sum()),
                "n_bars_in_fold": n_bars,
                "entry_date": index[cur_entry],
            })
            cur_entry = -1

    # Open at EOM: position never returned to 0
    if cur_entry >= 0:
        trades.append({
            "entry_idx": cur_entry,
            "exit_idx": n_bars - 1,
            "duration_bars": n_bars - 1 - cur_entry,
            "exit_code": 0,
            "exit_type": "open_at_eom",
            "net_pnl": float(pnl_net[cur_entry:].sum()),
            "n_bars_in_fold": n_bars,
            "entry_date": index[cur_entry],
        })

    return trades


def main() -> int:
    print(f"=== Entry-day distribution audit | fold {FOLD_N} ===\n", flush=True)

    sched = [s for s in FOLD_SCHEDULE if s[0] == FOLD_N][0]
    fold_n, fs, fe, tm = sched
    print(f"Trading month: {tm} | formation {fs} .. {fe}\n", flush=True)

    print("Loading daily cache...", flush=True)
    cache_daily = _load_all_daily(DATA_DAILY)
    print(f"  {len(cache_daily)} tickers loaded\n", flush=True)

    print("Loading dynamic cost data...", flush=True)
    cost_data = load_cost_data()
    print(f"  {len(cost_data.daily_spread)} spread rows, "
          f"{len(cost_data.kappa_map)} kappa entries\n", flush=True)

    print(f"Running fold {FOLD_N} (V4 defaults, β-cap=5, dynamic cost)...\n",
          flush=True)
    res = run_fold_v4(
        fold_n=fold_n, formation_start=fs, formation_end=fe,
        trading_month=tm, cache_daily=cache_daily,
        out_dir=Path("results/v4/audits"),
        cost_data=cost_data,
    )
    if res is None:
        print("Fold returned None -- abort")
        return 1

    pair_results = res["pair_results"]
    print(f"\nExtracting trades from {len(pair_results)} pairs...\n", flush=True)

    all_trades: list[dict] = []
    for (ta, tb), pdf in pair_results.items():
        for t in extract_trades(pdf):
            t["pair"] = f"{ta}/{tb}"
            all_trades.append(t)

    if not all_trades:
        print("No trades found -- abort")
        return 1

    df = pd.DataFrame(all_trades)
    # entry-day-in-window is 1-indexed
    df["entry_day"] = df["entry_idx"] + 1
    df["exit_day"] = df["exit_idx"] + 1
    df["days_to_eom_from_entry"] = df["n_bars_in_fold"] - df["entry_day"]
    n_bars = int(df["n_bars_in_fold"].iloc[0])

    print(f"Total trades        : {len(df)}")
    print(f"Trading window      : {n_bars} bars")
    print(f"Mean entry day      : {df['entry_day'].mean():.1f}")
    print(f"Median entry day    : {df['entry_day'].median():.1f}")
    print(f"Mean days-to-EOM    : {df['days_to_eom_from_entry'].mean():.1f}")
    print()

    # Sanity: invariant n_trades = zero + hard + EOM should hold
    counts = df["exit_type"].value_counts().to_dict()
    print(f"Exit breakdown      : {counts}")
    print()

    # --- Bucket: Early / Mid / Late ---
    t1 = n_bars / 3.0
    t2 = 2.0 * n_bars / 3.0

    def bucket(d: float) -> str:
        if d <= t1:
            return f"Early (1-{int(round(t1))})"
        if d <= t2:
            return f"Mid ({int(round(t1))+1}-{int(round(t2))})"
        return f"Late ({int(round(t2))+1}-{n_bars})"

    df["bucket"] = df["entry_day"].apply(bucket)
    order = [bucket(1), bucket(round(t1) + 1), bucket(round(t2) + 1)]

    print("=== Exit type mix by entry bucket (% of trades in bucket) ===")
    cross = pd.crosstab(
        df["bucket"], df["exit_type"], margins=True, normalize="index",
    ).mul(100).round(1)
    cross = cross.reindex(order + ["All"])
    print(cross.to_string())
    print()

    print("=== P&L by entry bucket ===")
    grp = df.groupby("bucket")["net_pnl"].agg(
        n_trades="count", sum_pnl="sum", mean_pnl="mean", median_pnl="median",
    ).round(1)
    grp = grp.reindex(order)
    print(grp.to_string())
    print()

    print("=== Zero-cross rate vs entry day ===")
    df["is_zero"] = (df["exit_code"] == 1).astype(int)
    df["is_eom"] = (df["exit_type"] == "open_at_eom").astype(int)
    by_day = df.groupby("entry_day").agg(
        n=("is_zero", "size"),
        zero_cross_rate=("is_zero", "mean"),
        eom_rate=("is_eom", "mean"),
        mean_pnl=("net_pnl", "mean"),
    ).round(3)
    print(by_day.to_string())
    print()

    print("=== Summary inference for cause B ===")
    late_frac = (df["entry_day"] > t2).mean() * 100
    mid_frac = ((df["entry_day"] > t1) & (df["entry_day"] <= t2)).mean() * 100
    early_frac = (df["entry_day"] <= t1).mean() * 100

    zc_early = df.loc[df["entry_day"] <= t1, "is_zero"].mean() * 100
    zc_mid = df.loc[(df["entry_day"] > t1) & (df["entry_day"] <= t2), "is_zero"].mean() * 100
    zc_late = df.loc[df["entry_day"] > t2, "is_zero"].mean() * 100

    print(f"  Entry distribution : Early {early_frac:.1f}% | Mid {mid_frac:.1f}% | Late {late_frac:.1f}%")
    print(f"  Zero-cross rate    : Early {zc_early:.1f}% | Mid {zc_mid:.1f}% | Late {zc_late:.1f}%")
    if zc_early > zc_late + 5:
        print(f"  -> Cause B CONFIRMED: late entries revert less often "
              f"({zc_early:.1f}% -> {zc_late:.1f}%, drop of {zc_early-zc_late:.1f}pp)")
    elif zc_early >= zc_late:
        print(f"  -> Cause B WEAK: small drop ({zc_early:.1f}% -> {zc_late:.1f}%)")
    else:
        print(f"  -> Cause B NOT supported: late entries revert MORE often, "
              f"hypothesis wrong")

    out_path = Path("results/v4/audit_entry_day_distribution.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["n_bars_in_fold"]).to_csv(out_path, index=False)
    print(f"\nFull trade log: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
