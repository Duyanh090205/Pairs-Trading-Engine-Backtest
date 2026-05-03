"""
run_phase4_final.py — Phase 4 Analytics on Option A + CORR25 Final Pipeline Results

Reads:  results/metrics/final/fold_metrics.csv  (23 completed folds)
        results/logs/final/fold_NN_audit.txt

Writes: results/metrics/final/phase4/  — all §1/§2/§5/§6/§7/§8/§9/§10/§11 CSVs
        results/figures/final/          — all figures

Does NOT overwrite:
  results/metrics/   (baseline Phase 3 outputs)
  results/figures/   (baseline Phase 4 figures)

Skipped (config-independent, unchanged from baseline run):
  §3  pair persistence   — re-run with --persistence if needed
  §4  volume strat       — re-run with --volume-strat if needed
  §12 universe counts    — same Phase 1 pairs regardless of downstream filters

Usage:
    python run_phase4_final.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── 1. Add src to path (must happen before any src import) ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4_final")

# ── 2. Monkey-patch orchestrator BEFORE importing sub-modules ───────────────
#
# Phase 4 sub-modules (regime.py, overfitting.py, sensitivity.py) do:
#   from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR, ...
# at module-level on import.  If we patch orchestrator FIRST and import
# sub-modules AFTER, their module-level bindings pick up the patched paths.

import src.phase4_defense.orchestrator as _orch  # noqa: E402

_FINAL_METRICS = Path("results/metrics/final/phase4")
_FINAL_FIGURES = Path("results/figures/final")
_FINAL_LOGS    = Path("results/logs/final")
_FINAL_SRC_CSV = Path("results/metrics/final/fold_metrics.csv")

# Override module-level path attributes
_orch._METRICS_DIR = _FINAL_METRICS
_orch._FIGURES_DIR = _FINAL_FIGURES
_orch._LOGS_DIR    = _FINAL_LOGS

# Create output directories before sub-module imports (they call mkdir at import time)
_FINAL_METRICS.mkdir(parents=True, exist_ok=True)
_FINAL_FIGURES.mkdir(parents=True, exist_ok=True)


def _load_final_fold_metrics() -> pd.DataFrame:
    """Load fold_metrics from final pipeline run and attach regime labels."""
    df = pd.read_csv(_FINAL_SRC_CSV)
    df["regime"] = df["fold"].map(_orch.FOLD_TO_REGIME)
    return df


# Replace the orchestrator's loader so sub-modules calling load_fold_metrics()
# with no arguments get the final-pipeline data automatically.
_orch.load_fold_metrics = _load_final_fold_metrics

# Patch load_equity to read from results/metrics/final/ (not the phase4/ subdir).
# DSR in overfitting.py calls get_daily_returns -> load_equity, which uses
# _METRICS_DIR / "fold{N:02d}_equity.parquet".  Since _METRICS_DIR is now the
# phase4/ subdir, we override load_equity to point at the actual equity location.
_FINAL_EQUITY_DIR = Path("results/metrics/final")

def _load_final_equity(fold_n: int):
    path = _FINAL_EQUITY_DIR / f"fold{fold_n:02d}_equity.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

_orch.load_equity = _load_final_equity

# ── 3. NOW import sub-modules (they see patched _METRICS_DIR / _FIGURES_DIR) ─
from src.phase4_defense.regime import run_regime_analysis, print_regime_report  # noqa: E402
from src.phase4_defense.overfitting import (  # noqa: E402
    run_overfitting_diagnostics,
    print_overfitting_report,
)
from src.phase4_defense.sensitivity import (  # noqa: E402
    run_oat_sensitivity,
    compute_exit_reasons,
    print_oat_report,
)


# ── Helper runners (same logic as run_phase4.py, paths from patched orchestrator)

def _run_sharpe_distribution(fold_metrics: pd.DataFrame) -> None:
    dist = fold_metrics[["fold", "trading_month", "sharpe", "max_dd", "cagr", "calmar"]].copy()
    dist.to_csv(_FINAL_METRICS / "sharpe_distribution.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.hist(fold_metrics["sharpe"], bins=15, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(fold_metrics["sharpe"].median(), color="red",
                   linewidth=1.5, linestyle="--",
                   label=f"Median={fold_metrics['sharpe'].median():.2f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Fold Count")
        ax.set_title(f"45-Fold Sharpe Distribution (Option A+CORR25)\n"
                     f"N={len(fold_metrics)}, mean={fold_metrics['sharpe'].mean():.2f}, "
                     f"% pos={(fold_metrics['sharpe'] > 0).mean():.0%}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        months = fold_metrics["trading_month"].tolist()
        x = range(len(months))
        ax2.bar(x, fold_metrics["sharpe"],
                color=["#2ca02c" if s > 0 else "#d62728" for s in fold_metrics["sharpe"]],
                alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(list(x)[::4])
        ax2.set_xticklabels(months[::4], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe per Fold (chronological) — Option A+CORR25")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "sharpe_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Sharpe distribution figure saved → %s", _FINAL_FIGURES / "sharpe_hist.png")
    except ImportError:
        log.warning("matplotlib not available — skipping figure")


def _run_cost_decomposition(fold_metrics: pd.DataFrame) -> None:
    cols = ["fold", "trading_month", "sharpe",
            "cost_commission", "cost_borrow", "cost_rebalance", "n_trades"]
    cost = fold_metrics[[c for c in cols if c in fold_metrics.columns]].copy()
    cost["cost_total"] = (
        cost.get("cost_commission", 0)
        + cost.get("cost_borrow", 0)
        + cost.get("cost_rebalance", 0)
    )
    cost.to_csv(_FINAL_METRICS / "cost_decomp.csv", index=False)

    total_capital = 1_000_000.0
    print("\n  Cost Decomposition ($ totals, 23 completed folds):")
    for col in ["cost_commission", "cost_borrow", "cost_rebalance", "cost_total"]:
        if col in cost:
            tot = cost[col].sum()
            print(f"    {col:<20}: ${tot:>12,.0f}  "
                  f"({tot / total_capital / len(cost) * 100:.1f} bps/fold/capital)")


def _run_delta_trajectory(fold_metrics: pd.DataFrame) -> None:
    delta_df = fold_metrics[["fold", "trading_month", "delta"]].copy()
    delta_df.to_csv(_FINAL_METRICS / "delta_trajectory.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.semilogy(range(len(delta_df)), delta_df["delta"], "o-",
                    markersize=4, linewidth=1.5, color="#1f77b4")
        ax.set_xticks(range(0, len(delta_df), max(1, len(delta_df) // 8)))
        ax.set_xticklabels(
            delta_df["trading_month"].iloc[::max(1, len(delta_df) // 8)].tolist(),
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("Selected δ (log scale)")
        ax.set_title("Kalman δ Trajectory — Option A+CORR25 (fixed δ=1e-7)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "delta_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _extract_nc_latency(fold_metrics: pd.DataFrame) -> None:
    # NC bootstrap (§7)
    nc_cols = ["fold", "trading_month", "nc_threshold", "nc_pass", "sharpe"]
    nc_df = fold_metrics[[c for c in nc_cols if c in fold_metrics.columns]].copy()
    nc_df.to_csv(_FINAL_METRICS / "nc_bootstrap.csv", index=False)
    n_pass = int(nc_df["nc_pass"].sum()) if "nc_pass" in nc_df.columns else 0
    n_total = int(nc_df["nc_pass"].notna().sum()) if "nc_pass" in nc_df.columns else len(nc_df)
    print(f"\n  NC bootstrap: {n_pass}/{n_total} folds pass  "
          f"(Primary > NC threshold at same eos_flatten=False)")

    # Latency (§8)
    lat_cols = ["fold", "trading_month", "t1_sharpe", "t5_sharpe", "t10_sharpe", "latency_pass"]
    lat_df = fold_metrics[[c for c in lat_cols if c in fold_metrics.columns]].copy()
    lat_df.to_csv(_FINAL_METRICS / "latency_decay.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for col, lag, color in [
            ("t1_sharpe",  "t+1",  "#2ca02c"),
            ("t5_sharpe",  "t+5",  "#ff7f0e"),
            ("t10_sharpe", "t+10", "#d62728"),
        ]:
            if col in lat_df.columns:
                valid = lat_df[col].dropna()
                ax.plot(range(len(valid)), valid.values, "o-",
                        markersize=3, linewidth=1.2, alpha=0.8,
                        label=f"Lag {lag} (mean={valid.mean():.2f})",
                        color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold (completed only)")
        ax.set_ylabel("Sharpe")
        ax.set_title("Latency Sweep — Option A+CORR25")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "latency_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _print_exit_summary(exit_df: pd.DataFrame) -> None:
    tot = exit_df["n_trades_parsed"].sum()
    if tot == 0:
        print("  No trade records parsed from audit logs.")
        return
    eos = exit_df["n_eos"].sum()
    zc  = exit_df["n_zero_cross"].sum()
    sl  = exit_df.get("n_sl", pd.Series(0, index=exit_df.index)).sum()

    print("\n  Exit reasons (aggregate, Option A+CORR25):")
    print(f"    EOS exits:       {int(eos):6d}  ({eos / tot:.1%})")
    print(f"    Zero-cross:      {int(zc):6d}  ({zc / tot:.1%})")
    if sl > 0:
        print(f"    Stop-loss:       {int(sl):6d}  ({sl / tot:.1%})")
    if "avg_net_eos" in exit_df.columns:
        print(f"    Avg net EOS:     {exit_df['avg_net_eos'].mean():.1f} bps")
    if "avg_net_zc" in exit_df.columns:
        print(f"    Avg net ZC:      {exit_df['avg_net_zc'].mean():.1f} bps")


def _check_outputs() -> None:
    required_metrics = [
        "sharpe_distribution.csv",    # §1
        "regime_sharpes.csv",          # §2
        "overfitting_diagnostics.csv", # §5
        "oat_sensitivity.csv",         # §6
        "nc_bootstrap.csv",            # §7
        "latency_decay.csv",           # §8
        "exit_reasons.csv",            # §9
        "cost_decomp.csv",             # §10
        "delta_trajectory.csv",        # §11
    ]
    required_figures = [
        "sharpe_hist.png",
        "regime_bar.png",
        "delta_traj.png",
        "latency_curve.png",
        "oat_grid.png",
    ]

    print()
    all_ok = True
    for fname in required_metrics:
        path = _FINAL_METRICS / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/metrics/final/phase4/{fname:<35} {status}")

    for fname in required_figures:
        path = _FINAL_FIGURES / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/figures/final/{fname:<40} {status}")

    print()
    skipped = ["§3 pair_persistence.csv", "§4 volume_strat.csv", "§12 universe_counts.csv"]
    print(f"  Skipped (config-independent): {', '.join(skipped)}")

    if all_ok:
        log.info("All final-pipeline Phase 4 outputs present.")
    else:
        log.warning("Some outputs missing — check warnings above.")


# ── Main ───────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not _FINAL_SRC_CSV.exists():
        log.error("Source not found: %s  — run run_final_pipeline.py first.", _FINAL_SRC_CSV)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 4 — Option A+CORR25 Final Pipeline Analytics")
    log.info("  Input:  %s", _FINAL_SRC_CSV)
    log.info("  Output: %s", _FINAL_METRICS)
    log.info("=" * 60)

    fold_metrics = _load_final_fold_metrics()
    log.info("Loaded %d completed folds", len(fold_metrics))
    _orch.print_fold_summary(fold_metrics)

    # ── §1: Sharpe distribution ────────────────────────────────────────────
    log.info("\n[§1] Sharpe distribution")
    _run_sharpe_distribution(fold_metrics)

    # ── §2: Regime partition ───────────────────────────────────────────────
    log.info("\n[§2] Regime partition")
    regime_df = run_regime_analysis(fold_metrics, save=True)
    print_regime_report(regime_df)

    # ── §5: Overfitting diagnostics ────────────────────────────────────────
    log.info("\n[§5] Overfitting diagnostics (DSR + PBO)")
    overfit_df = run_overfitting_diagnostics(fold_metrics, save=True)
    print_overfitting_report(overfit_df)

    # ── §6: OAT sensitivity (analytical only) ─────────────────────────────
    log.info("\n[§6] OAT sensitivity (analytical)")
    oat_df = run_oat_sensitivity(
        fold_metrics,
        run_structural=args.structural_oat if hasattr(args, "structural_oat") else False,
        structural_max_folds=5,
        save=True,
    )
    print_oat_report(oat_df)

    # ── §9: Exit reason breakdown from final audit logs ────────────────────
    log.info("\n[§9] Exit reason breakdown")
    exit_df = compute_exit_reasons(logs_dir=_FINAL_LOGS)
    if not exit_df.empty:
        _print_exit_summary(exit_df)
    else:
        log.warning("[§9] No exit records parsed from %s", _FINAL_LOGS)

    # ── §10: Cost decomposition ────────────────────────────────────────────
    log.info("\n[§10] Cost decomposition")
    _run_cost_decomposition(fold_metrics)

    # ── §11: Delta trajectory ──────────────────────────────────────────────
    log.info("\n[§11] Delta trajectory")
    _run_delta_trajectory(fold_metrics)

    # ── §7 + §8: NC bootstrap + latency ───────────────────────────────────
    log.info("\n[§7+8] NC bootstrap + latency — extracting from fold_metrics")
    _extract_nc_latency(fold_metrics)

    # ── §3: Pair persistence (optional) ───────────────────────────────────
    if getattr(args, "persistence", False):
        log.info("\n[§3] Pair persistence (re-running Johansen, ~30 min)")
        from src.phase4_defense.persistence import run_persistence
        persist_df = run_persistence(save=True)
        log.info("Persistence complete: %d fold points", len(persist_df))
    else:
        log.info("\n[§3] Pair persistence skipped (pass --persistence to run)")

    # ── §4: Volume stratification (optional) ──────────────────────────────
    if getattr(args, "volume_strat", False):
        log.info("\n[§4] Volume stratification (~15 min)")
        from src.phase4_defense.volume_strat import run_volume_strat, print_volume_report
        vol_fold, vol_pairs = run_volume_strat(save=True)
        print_volume_report(vol_fold, vol_pairs)
    else:
        log.info("\n[§4] Volume stratification skipped (pass --volume-strat to run)")

    # ── Final checklist ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Phase 4 final analytics complete. Output files:")
    _check_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 analytics on Option A+CORR25 final pipeline results"
    )
    parser.add_argument("--structural-oat", action="store_true",
                        help="Run structural OAT variations (slow, ~20 min)")
    parser.add_argument("--persistence",    action="store_true",
                        help="Run P_2022 pair persistence (~30 min)")
    parser.add_argument("--volume-strat",   action="store_true",
                        help="Run volume stratification (~15 min)")
    args = parser.parse_args()
    main(args)
