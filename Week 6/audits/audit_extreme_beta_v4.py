"""
Audit the beta distribution across all 39 V4 folds — quantify extreme-beta
contamination scope.

For each fold, run discovery, save (ticker_a, ticker_b, beta_pca, johansen_pval,
half_life_days, alpha_pca) for all surviving pairs. Then:

  1. Distribution of |beta| across folds (percentiles + bucket counts)
  2. Per-fold count of extreme-beta pairs (|β| > 3, > 5, > 10, > 100)
  3. Identify which pairs would survive the engine's concentration cap
  4. Flag folds with most extreme contamination
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

HL_MAX = 30.0
HL_MIN = 5.0
_MAX_PAIRS_PER_TICKER = 5


def apply_ticker_concentration_cap(pairs_df: pd.DataFrame,
                                   max_pairs_per_ticker: int = _MAX_PAIRS_PER_TICKER) -> pd.DataFrame:
    """Inlined copy of engine.phase2_execution.engine.apply_ticker_concentration_cap
    (avoid that module's broken import of deleted z_strategies)."""
    if pairs_df.empty:
        return pairs_df
    ranked = pairs_df.sort_values("johansen_pval").reset_index(drop=True)
    ticker_count: dict[str, int] = {}
    keep_idx: list[int] = []
    for idx, row in ranked.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        ca = ticker_count.get(ta, 0)
        cb = ticker_count.get(tb, 0)
        if ca < max_pairs_per_ticker and cb < max_pairs_per_ticker:
            keep_idx.append(idx)
            ticker_count[ta] = ca + 1
            ticker_count[tb] = cb + 1
    return ranked.loc[keep_idx].reset_index(drop=True)


def main() -> int:
    print("=== V4 extreme-beta audit across all 39 folds ===\n")
    t0 = time.time()
    cache = _load_all_daily(DATA_DAILY)
    print(f"cache loaded ({time.time()-t0:.1f}s)\n")

    all_pairs_rows = []
    for fold_n, fs, fe, tm in FOLD_SCHEDULE:
        t_fold = time.time()
        formation = _slice_daily(cache, fs, fe)
        if not formation:
            print(f"  Fold {fold_n}: no formation data, skip")
            continue
        try:
            pairs_df, factor_state = discovery_daily.run(formation, hl_min=HL_MIN, hl_max=HL_MAX)
        except Exception as e:
            print(f"  Fold {fold_n}: discovery failed: {e}")
            continue
        if pairs_df.empty:
            print(f"  Fold {fold_n} [{tm}]: 0 pairs after discovery, skip")
            continue

        # Concentration cap (only these would actually trade)
        pairs_capped = apply_ticker_concentration_cap(pairs_df)
        traded_set = set(zip(pairs_capped["ticker_a"], pairs_capped["ticker_b"]))

        for _, row in pairs_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            beta = float(row["beta_pca"])
            all_pairs_rows.append({
                "fold": fold_n,
                "trading_month": tm,
                "ticker_a": ta, "ticker_b": tb,
                "beta_pca": beta,
                "abs_beta": abs(beta),
                "alpha_pca": float(row["alpha_pca"]),
                "johansen_pval": float(row["johansen_pval"]),
                "half_life_days": float(row["half_life_days"]),
                "passes_concentration_cap": (ta, tb) in traded_set,
            })

        n_total = len(pairs_df)
        n_traded = len(pairs_capped)
        beta_abs = pairs_df["beta_pca"].abs()
        n_ext5  = int((beta_abs > 5).sum())
        n_ext10 = int((beta_abs > 10).sum())
        n_ext50 = int((beta_abs > 50).sum())
        print(f"  Fold {fold_n:02d} [{tm}]: discovery={n_total:>3}  traded={n_traded:>2}  "
              f"|β|>5: {n_ext5:>2}  |β|>10: {n_ext10:>2}  |β|>50: {n_ext50:>2}  "
              f"({time.time()-t_fold:.0f}s)")

    df = pd.DataFrame(all_pairs_rows)
    out_path = WEEK6 / "results" / "v4" / "extreme_beta_audit.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df):,} pair-fold rows)\n")

    # -------- analysis --------
    print("="*80)
    print("OVERALL BETA DISTRIBUTION across all folds (all surviving pairs)")
    print("="*80)
    print(f"  Total pair-fold rows: {len(df):,}")
    print(f"  abs(β) percentiles: ")
    for p in [25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        print(f"    p{p:>4.1f}: {df['abs_beta'].quantile(p/100):.3f}")

    print()
    print("="*80)
    print("EXTREME-β BUCKETS")
    print("="*80)
    buckets = [(1, 3), (3, 5), (5, 10), (10, 50), (50, 100), (100, 1000), (1000, 1e9)]
    for lo, hi in buckets:
        mask = (df['abs_beta'] >= lo) & (df['abs_beta'] < hi)
        n_all = mask.sum()
        n_traded = (mask & df['passes_concentration_cap']).sum()
        pct_all = 100 * n_all / len(df)
        print(f"  |β| in [{lo:>6.1f}, {hi:>9.1f}): "
              f"{n_all:>5,} pairs ({pct_all:>5.2f}%) -- of which {n_traded:>4} would trade after cap")

    print()
    print("="*80)
    print("PER-FOLD COUNT OF EXTREME pairs that PASS concentration cap (would trade)")
    print("="*80)
    extreme = df[df['abs_beta'] > 5].copy()
    if len(extreme) > 0:
        per_fold = extreme[extreme['passes_concentration_cap']].groupby('fold').agg(
            trading_month=('trading_month', 'first'),
            n_ext_traded=('abs_beta', 'count'),
            max_abs_beta=('abs_beta', 'max'),
            example_pair=('ticker_a', lambda x: x.iloc[0]),
        ).sort_values('n_ext_traded', ascending=False)
        print(per_fold.head(15).to_string(float_format=lambda x: f"{x:.2f}"))

    print()
    print("="*80)
    print("MOST EXTREME pairs (top 20 by |β|, traded or not)")
    print("="*80)
    top_ext = df.nlargest(20, 'abs_beta')[
        ['fold', 'trading_month', 'ticker_a', 'ticker_b', 'beta_pca', 'alpha_pca',
         'johansen_pval', 'half_life_days', 'passes_concentration_cap']
    ]
    print(top_ext.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Verify GEHC/WMT in F26
    print()
    print("="*80)
    print("VERIFY: GEHC/WMT in fold 26")
    print("="*80)
    gw = df[(df['fold'] == 26) & (
        ((df['ticker_a'] == 'GEHC') & (df['ticker_b'] == 'WMT')) |
        ((df['ticker_a'] == 'WMT') & (df['ticker_b'] == 'GEHC'))
    )]
    if len(gw) > 0:
        print(gw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  not found in fold 26")

    return 0


if __name__ == "__main__":
    sys.exit(main())
