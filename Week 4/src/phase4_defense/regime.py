"""
Phase 4b — Regime Partition Analysis

Partitions fold results into 4 market regimes and computes per-regime
Sharpe statistics. Outputs:
  results/metrics/regime_sharpes.csv
  results/figures/regime_bar.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    REGIME_MAP,
    load_fold_metrics,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_regime_analysis(
    fold_metrics: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute per-regime Sharpe statistics.

    Returns DataFrame with columns:
      [regime, n_folds, n_completed, mean_sharpe, median_sharpe,
       iqr_25, iqr_75, pct_positive, min_sharpe, max_sharpe,
       mean_calmar, mean_cagr, mean_win_rate]
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    rows = []
    for regime, folds_in in REGIME_MAP.items():
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)].copy()
        n_completed = len(sub)
        n_total     = len(folds_in)

        if n_completed == 0:
            rows.append({
                "regime": regime,
                "n_folds": n_total,
                "n_completed": 0,
                **{k: np.nan for k in [
                    "mean_sharpe","median_sharpe","iqr_25","iqr_75",
                    "pct_positive","min_sharpe","max_sharpe",
                    "mean_calmar","mean_cagr","mean_win_rate",
                ]},
            })
            continue

        sharpes = sub["sharpe"].values
        rows.append({
            "regime":       regime,
            "n_folds":      n_total,
            "n_completed":  n_completed,
            "mean_sharpe":  float(np.mean(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "iqr_25":       float(np.percentile(sharpes, 25)),
            "iqr_75":       float(np.percentile(sharpes, 75)),
            "pct_positive": float(np.mean(sharpes > 0)),
            "min_sharpe":   float(np.min(sharpes)),
            "max_sharpe":   float(np.max(sharpes)),
            "mean_calmar":  float(sub["calmar"].mean()) if "calmar" in sub else np.nan,
            "mean_cagr":    float(sub["cagr"].mean())   if "cagr"   in sub else np.nan,
            "mean_win_rate":float(sub["win_rate"].mean()) if "win_rate" in sub else np.nan,
        })

    result = pd.DataFrame(rows)

    if save:
        out = _METRICS_DIR / "regime_sharpes.csv"
        result.to_csv(out, index=False)
        log.info("Regime Sharpes saved → %s", out)
        _save_figure(result, fold_metrics)

    return result


def _save_figure(regime_df: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    """Save regime bar chart and Sharpe distribution histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available — skipping regime figures")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: per-regime mean ± IQR bar chart ---
    ax = axes[0]
    regimes      = regime_df["regime"].tolist()
    short_labels = ["Bear\n2022", "Bull\n2023", "Bull\n2024", "Bull\n2025-Q1"]
    means        = regime_df["mean_sharpe"].values
    p25          = regime_df["iqr_25"].values
    p75          = regime_df["iqr_75"].values
    n_comp       = regime_df["n_completed"].values

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]
    x = np.arange(len(regimes))
    bars = ax.bar(x, means, color=colors, alpha=0.75, width=0.55)
    ax.errorbar(x, means,
                yerr=[means - p25, p75 - means],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(short_labels, n_comp)],
                       fontsize=9)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Mean Sharpe by Regime (bars = IQR)")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: full Sharpe distribution histogram ---
    ax2 = axes[1]
    for (regime, folds_in), color in zip(REGIME_MAP.items(), colors):
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)]["sharpe"]
        if len(sub) > 0:
            ax2.hist(sub.values, bins=8, alpha=0.55, color=color,
                     label=regime.split(" ")[0] + " " + regime.split(" ")[1])
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Fold Count")
    ax2.set_title("Fold Sharpe Distribution by Regime")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _FIGURES_DIR / "regime_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Regime figure saved → %s", out)


def print_regime_report(regime_df: pd.DataFrame | None = None) -> None:
    """Pretty-print regime stats to console."""
    if regime_df is None:
        regime_df = run_regime_analysis(save=False)

    print("\n=== Regime Partition Analysis ===")
    print(f"{'Regime':<30} {'N':>4} {'Done':>4} "
          f"{'Mean':>7} {'Median':>7} {'IQR':>15} "
          f"{'%Pos':>6} {'WinRate':>8}")
    print("-" * 90)

    for _, row in regime_df.iterrows():
        iqr_str = f"[{row['iqr_25']:.2f}, {row['iqr_75']:.2f}]"
        wr = f"{row['mean_win_rate']:.1%}" if not np.isnan(row['mean_win_rate']) else "  n/a"
        print(f"{row['regime']:<30} {row['n_folds']:>4} {row['n_completed']:>4} "
              f"{row['mean_sharpe']:>7.2f} {row['median_sharpe']:>7.2f} "
              f"{iqr_str:>15} {row['pct_positive']:>6.0%} {wr:>8}")

    print("\n  NOTE: Late Bear 2022 N=6 is underpowered for strong inference.")
