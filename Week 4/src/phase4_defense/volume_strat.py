"""
Phase 4d — Volume Stratification (Hussein-inspired)

For each fold, divides the survivor universe into share-volume ADV tertiles
(T1=low, T2=mid, T3=high) and tags surviving pairs by same-tertile bucket.
Compares distribution of cointegrated pairs across T1/T2/T3.

NOTE: This is documented as an OUT-OF-DOMAIN extension of Hussein et al.,
not a replication. Cross-tertile pairs are deferred to Week 5+.

Outputs:
  results/metrics/volume_strat.csv
  results/metrics/volume_strat_pairs.csv   (pair-level ADV tags)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_phase1_pairs,
    load_fold_metrics,
    _METRICS_DIR,
    _FIGURES_DIR,
)
from src.utils.io import read_5min

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _compute_adv(tickers: list[str], form_start: str, form_end: str) -> dict[str, float]:
    """
    Compute average daily share volume for each ticker in the formation window.
    Uses 5-min data (volume column) resampled to daily sums then averaged.
    """
    adv: dict[str, float] = {}
    for ticker in tickers:
        try:
            df = read_5min(ticker)
            if df is None or df.empty or "volume" not in df.columns:
                continue
            mask = (df.index >= form_start) & (df.index <= form_end)
            sub = df.loc[mask, "volume"].dropna()
            if len(sub) < 10:
                continue
            daily_vol = sub.resample("B").sum()
            adv[ticker] = float(daily_vol.mean())
        except Exception:
            pass
    return adv


def _assign_tertiles(adv: dict[str, float]) -> dict[str, str]:
    """Assign T1/T2/T3 tertile label per ticker based on ADV."""
    if not adv:
        return {}
    series = pd.Series(adv)
    t1_thresh = float(series.quantile(1 / 3))
    t2_thresh = float(series.quantile(2 / 3))
    result: dict[str, str] = {}
    for ticker, val in adv.items():
        if val <= t1_thresh:
            result[ticker] = "T1"
        elif val <= t2_thresh:
            result[ticker] = "T2"
        else:
            result[ticker] = "T3"
    return result


def run_volume_strat(
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tag each surviving pair with its ADV tertile bucket per fold.
    Only same-tertile pairs are tagged (T1-T1, T2-T2, T3-T3).
    Cross-tertile pairs get bucket='cross'.

    Returns
    -------
    fold_summary : DataFrame per fold — pair count by tertile
    pair_tags    : DataFrame per pair per fold — with tertile bucket assigned
    """
    fold_rows:  list[dict] = []
    pair_rows:  list[dict] = []

    fold_metrics = load_fold_metrics()
    completed_folds = set(fold_metrics["fold"].tolist())

    for spec in FOLD_SCHEDULE:
        fold_n = spec.fold_n

        pairs_df = load_phase1_pairs(fold_n)
        if pairs_df is None or pairs_df.empty:
            continue

        # All unique tickers in this fold's survivor universe
        tickers = list(set(
            pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
        ))

        log.info("Volume strat fold %d: %d tickers, %d pairs",
                 fold_n, len(tickers), len(pairs_df))

        # Compute ADV for each ticker over formation window
        adv = _compute_adv(tickers, spec.form_start, spec.form_end)

        if not adv:
            log.warning("Fold %d: no ADV data available", fold_n)
            continue

        tertile_map = _assign_tertiles(adv)

        n_t1 = n_t2 = n_t3 = n_cross = 0

        for _, row in pairs_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            ta_tertile = tertile_map.get(ta)
            tb_tertile = tertile_map.get(tb)

            if ta_tertile is None or tb_tertile is None:
                bucket = "unknown"
            elif ta_tertile == tb_tertile:
                bucket = ta_tertile  # T1, T2, or T3
                if bucket == "T1":   n_t1 += 1
                elif bucket == "T2": n_t2 += 1
                else:                n_t3 += 1
            else:
                bucket = "cross"
                n_cross += 1

            pair_rows.append({
                "fold":          fold_n,
                "trading_month": spec.trading_month,
                "ticker_a":      ta,
                "ticker_b":      tb,
                "adv_a":         adv.get(ta, np.nan),
                "adv_b":         adv.get(tb, np.nan),
                "tertile_a":     ta_tertile,
                "tertile_b":     tb_tertile,
                "bucket":        bucket,
                "johansen_pval": row.get("johansen_pval", np.nan),
                "half_life_days":row.get("half_life_days", np.nan),
            })

        fold_rows.append({
            "fold":          fold_n,
            "trading_month": spec.trading_month,
            "n_pairs_total": len(pairs_df),
            "n_T1":          n_t1,
            "n_T2":          n_t2,
            "n_T3":          n_t3,
            "n_cross":       n_cross,
            "pct_T1":        n_t1 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_T2":        n_t2 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_T3":        n_t3 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_cross":     n_cross / len(pairs_df) if len(pairs_df) else np.nan,
            "has_fold_sharpe": fold_n in completed_folds,
        })

    fold_summary = pd.DataFrame(fold_rows)
    pair_tags    = pd.DataFrame(pair_rows)

    if save and not fold_summary.empty:
        fold_summary.to_csv(_METRICS_DIR / "volume_strat.csv", index=False)
        pair_tags.to_csv(_METRICS_DIR / "volume_strat_pairs.csv", index=False)
        log.info("Volume strat saved → results/metrics/volume_strat*.csv")
        _save_figure(fold_summary, pair_tags)

    return fold_summary, pair_tags


def print_volume_report(
    fold_summary: pd.DataFrame | None = None,
    pair_tags: pd.DataFrame | None = None,
) -> None:
    if fold_summary is None:
        fold_summary, pair_tags = run_volume_strat(save=False)

    print("\n=== Volume Stratification (Hussein-inspired) ===")
    print("NOTE: Out-of-domain extension, not Hussein replication.")
    print("      Cross-tertile pairs deferred to Week 5+.\n")

    # Aggregate across folds
    agg = fold_summary[["n_T1", "n_T2", "n_T3", "n_cross", "n_pairs_total"]].sum()
    tot = agg["n_pairs_total"]

    print(f"  Aggregate pair distribution across {len(fold_summary)} folds:")
    print(f"    T1-T1 (low vol):   {int(agg['n_T1']):5d}  ({agg['n_T1']/tot:.1%})")
    print(f"    T2-T2 (mid vol):   {int(agg['n_T2']):5d}  ({agg['n_T2']/tot:.1%})")
    print(f"    T3-T3 (high vol):  {int(agg['n_T3']):5d}  ({agg['n_T3']/tot:.1%})")
    print(f"    Cross-tertile:     {int(agg['n_cross']):5d}  ({agg['n_cross']/tot:.1%})")

    if pair_tags is not None and not pair_tags.empty:
        # HL comparison by tertile
        for bucket in ["T1", "T2", "T3"]:
            sub = pair_tags[pair_tags["bucket"] == bucket]["half_life_days"].dropna()
            if len(sub) > 0:
                print(f"\n  {bucket} half_life_days: "
                      f"median={sub.median():.2f}d  "
                      f"mean={sub.mean():.2f}d  N={len(sub)}")

    # Per-bucket trade metrics
    bucket_metrics = run_volume_strat_metrics(pair_tags=pair_tags)
    if not bucket_metrics.empty:
        print("\n  Per-bucket trade metrics (from pipeline run):")
        print(f"  {'Bucket':<8} {'N_pairs':>8} {'N_trades':>9} {'Win%':>7} {'AvgNetBps':>10} {'TotalPnL($k)':>13} {'MedianHL':>9}")
        for _, row in bucket_metrics.iterrows():
            print(
                f"  {row['bucket']:<8} {int(row['n_pairs']):>8} {int(row['n_trades']):>9} "
                f"{row['win_rate']*100:>6.1f}% {row['avg_net_bps']:>10.1f} "
                f"{row['total_pnl_k']:>12.1f}k {row['median_hl']:>8.2f}d"
            )
    else:
        print("\n  [Bucket metrics not available — run pipeline with updated code]")


def run_volume_strat_metrics(
    pair_tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join pair_tags with per-pair trade metrics saved by the pipeline runner.
    Computes per-bucket (T1/T2/T3/cross) breakdown of win_rate, avg_net_bps,
    n_trades, and total_pnl.

    Loads from results/metrics/pair_trade_metrics.csv (written by pipeline runner).
    Returns empty DataFrame if file not found.
    """
    metrics_path = _METRICS_DIR / "pair_trade_metrics.csv"
    if not metrics_path.exists():
        log.warning("pair_trade_metrics.csv not found — run pipeline with new code first")
        return pd.DataFrame()

    trade_m = pd.read_csv(metrics_path)

    if pair_tags is None:
        _, pair_tags = run_volume_strat(save=False)

    if pair_tags is None or pair_tags.empty:
        return pd.DataFrame()

    merged = pair_tags.merge(
        trade_m, on=["fold", "ticker_a", "ticker_b"], how="left"
    )

    rows = []
    for bucket in ["T1", "T2", "T3", "cross"]:
        sub = merged[merged["bucket"] == bucket].dropna(subset=["n_trades"])
        if sub.empty:
            continue
        n_pairs = len(sub)
        n_trades = int(sub["n_trades"].sum())
        rows.append({
            "bucket":        bucket,
            "n_pairs":       n_pairs,
            "n_trades":      n_trades,
            "win_rate":      float(sub["win_rate"].mean()),
            "win_rate_std":  float(sub["win_rate"].std()),
            "avg_net_bps":   float(sub["avg_net_bps"].mean()),
            "avg_net_bps_std": float(sub["avg_net_bps"].std()),
            "total_pnl_k":   float(sub["total_pnl"].sum() / 1000),
            "median_hl":     float(sub["half_life_days"].median()) if "half_life_days" in sub.columns else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(_METRICS_DIR / "volume_strat_bucket_metrics.csv", index=False)
        log.info("Bucket metrics saved → results/metrics/volume_strat_bucket_metrics.csv")
    return result


def _save_figure(fold_summary: pd.DataFrame, pair_tags: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Stacked bar: pair composition by tertile across folds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x  = np.arange(len(fold_summary))
    ax.bar(x, fold_summary["pct_T1"], label="T1 (low vol)", color="#d62728", alpha=0.8)
    ax.bar(x, fold_summary["pct_T2"], bottom=fold_summary["pct_T1"],
           label="T2 (mid vol)", color="#ff7f0e", alpha=0.8)
    ax.bar(x, fold_summary["pct_T3"],
           bottom=fold_summary["pct_T1"] + fold_summary["pct_T2"],
           label="T3 (high vol)", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Fraction of Pairs")
    ax.set_title("Pair Composition by ADV Tertile per Fold")
    ax.legend(fontsize=8)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(fold_summary["trading_month"].iloc[::5], rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # HL distribution by tertile
    ax2 = axes[1]
    colors = {"T1": "#d62728", "T2": "#ff7f0e", "T3": "#2ca02c"}
    for bucket, color in colors.items():
        sub = pair_tags[pair_tags["bucket"] == bucket]["half_life_days"].dropna()
        if len(sub) > 0:
            ax2.hist(sub.clip(upper=15).values, bins=20, alpha=0.55,
                     color=color, label=f"{bucket} (N={len(sub)})")
    ax2.set_xlabel("Half-Life (days, capped at 15)")
    ax2.set_ylabel("Pair Count")
    ax2.set_title("OU Half-Life Distribution by ADV Tertile")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _FIGURES_DIR / "volume_strat.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Volume strat figure saved → %s", out)
