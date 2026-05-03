"""
Phase 4 — Multi-Regime Defense Runner

Runs all analytical Phase 4 modules on existing pipeline results
and produces the 12 whitepaper output files.

Usage:
    python run_phase4.py                          # analytical only (fast ~2 min)
    python run_phase4.py --persistence            # + pair persistence (slow ~30 min)
    python run_phase4.py --volume-strat           # + volume stratification (~15 min)
    python run_phase4.py --structural-oat         # + structural OAT (~20 min)
    python run_phase4.py --all                    # everything (~60 min)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


def main(args: argparse.Namespace) -> None:
    from src.phase4_defense.orchestrator import (
        load_fold_metrics,
        print_fold_summary,
        _METRICS_DIR,
        _FIGURES_DIR,
    )

    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 4 — Multi-Regime Defense Analysis")
    log.info("=" * 60)

    # ── Load existing results ──────────────────────────────────────
    fold_metrics = load_fold_metrics()
    log.info("Loaded %d completed folds from fold_metrics.csv", len(fold_metrics))
    print_fold_summary(fold_metrics)

    # ── §1: 45-fold Sharpe distribution ───────────────────────────
    log.info("\n[§1] Sharpe distribution")
    _run_sharpe_distribution(fold_metrics)

    # ── §2: Regime partition ───────────────────────────────────────
    log.info("\n[§2] Regime partition")
    from src.phase4_defense.regime import run_regime_analysis, print_regime_report
    regime_df = run_regime_analysis(fold_metrics, save=True)
    print_regime_report(regime_df)

    # ── §5: Overfitting diagnostics ────────────────────────────────
    log.info("\n[§5] Overfitting diagnostics (DSR + PBO)")
    from src.phase4_defense.overfitting import run_overfitting_diagnostics, print_overfitting_report
    overfit_df = run_overfitting_diagnostics(fold_metrics, save=True)
    print_overfitting_report(overfit_df)

    # ── §6: OAT sensitivity (analytical) ──────────────────────────
    log.info("\n[§6] OAT sensitivity (analytical)")
    from src.phase4_defense.sensitivity import (
        run_oat_sensitivity, compute_exit_reasons, print_oat_report
    )
    oat_df = run_oat_sensitivity(
        fold_metrics,
        run_structural=args.structural_oat or args.all,
        structural_max_folds=10,
        save=True,
    )
    print_oat_report(oat_df)

    # ── §9: Exit reason breakdown ──────────────────────────────────
    log.info("\n[§9] Exit reason breakdown")
    exit_df = compute_exit_reasons()
    if not exit_df.empty:
        _print_exit_summary(exit_df)

    # ── §10: Cost decomposition ────────────────────────────────────
    log.info("\n[§10] Cost decomposition")
    _run_cost_decomposition(fold_metrics)

    # ── §11: Delta trajectory ──────────────────────────────────────
    log.info("\n[§11] Delta trajectory")
    _run_delta_trajectory(fold_metrics)

    # ── §12: Universe counts ───────────────────────────────────────
    log.info("\n[§12] Universe counts")
    _run_universe_counts()

    # ── §7 + §8: NC + latency (already in fold_metrics.csv) ───────
    log.info("\n[§7+8] NC bootstrap + latency — extracting from fold_metrics")
    _extract_nc_latency(fold_metrics)

    # ── §3: Pair persistence (optional — slow) ─────────────────────
    if args.persistence or args.all:
        log.info("\n[§3] Pair persistence (re-running Johansen, ~30 min)")
        from src.phase4_defense.persistence import run_persistence
        persist_df = run_persistence(save=True)
        log.info("Persistence complete: %d fold points", len(persist_df))
    else:
        log.info("\n[§3] Pair persistence skipped (pass --persistence to run)")

    # ── §4: Volume stratification (optional — slower) ─────────────
    if args.volume_strat or args.all:
        log.info("\n[§4] Volume stratification (~15 min)")
        from src.phase4_defense.volume_strat import run_volume_strat, print_volume_report
        vol_fold, vol_pairs = run_volume_strat(save=True)
        print_volume_report(vol_fold, vol_pairs)
    else:
        log.info("\n[§4] Volume stratification skipped (pass --volume-strat to run)")

    # ── Final checklist ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Phase 4 complete. Checking whitepaper output files:")
    _check_outputs()


# ---------------------------------------------------------------------------
# Helper runners for §1, §10-12, §7-8
# ---------------------------------------------------------------------------

def _run_sharpe_distribution(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    dist = fold_metrics[["fold", "trading_month", "sharpe", "max_dd", "cagr", "calmar"]].copy()
    dist.to_csv(_METRICS_DIR / "sharpe_distribution.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.hist(fold_metrics["sharpe"], bins=15, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(fold_metrics["sharpe"].median(), color="red",
                   linewidth=1.5, linestyle="--", label=f"Median={fold_metrics['sharpe'].median():.2f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Fold Count")
        ax.set_title(f"45-Fold Sharpe Distribution\n"
                     f"(N={len(fold_metrics)}, mean={fold_metrics['sharpe'].mean():.2f}, "
                     f"% pos={( fold_metrics['sharpe']>0).mean():.0%})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        months = fold_metrics["trading_month"].tolist()
        x = range(len(months))
        ax2.bar(x, fold_metrics["sharpe"], color=[
            "#2ca02c" if s > 0 else "#d62728" for s in fold_metrics["sharpe"]
        ], alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(list(x)[::6])
        ax2.set_xticklabels(months[::6], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe per Fold (chronological)")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "sharpe_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _run_cost_decomposition(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR

    cols = ["fold", "trading_month", "sharpe",
            "cost_commission", "cost_borrow", "cost_rebalance", "n_trades"]
    cost = fold_metrics[[c for c in cols if c in fold_metrics.columns]].copy()
    cost["cost_total"] = (
        cost.get("cost_commission", 0)
        + cost.get("cost_borrow", 0)
        + cost.get("cost_rebalance", 0)
    )
    cost.to_csv(_METRICS_DIR / "cost_decomp.csv", index=False)

    # Print summary
    print("\n  Cost Decomposition ($ totals across all folds):")
    total_capital = 1_000_000.0
    for col in ["cost_commission", "cost_borrow", "cost_rebalance", "cost_total"]:
        if col in cost:
            tot = cost[col].sum()
            print(f"    {col:<20}: ${tot:>15,.0f}  ({tot/total_capital/len(cost)*100:.1f} bps/fold/capital)")


def _run_delta_trajectory(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    delta_df = fold_metrics[["fold", "trading_month", "delta"]].copy()
    delta_df.to_csv(_METRICS_DIR / "delta_trajectory.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.semilogy(range(len(delta_df)), delta_df["delta"], "o-",
                    markersize=4, linewidth=1.5, color="#1f77b4")
        ax.set_xticks(range(0, len(delta_df), max(1, len(delta_df) // 8)))
        ax.set_xticklabels(
            delta_df["trading_month"].iloc[::max(1, len(delta_df) // 8)],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("Selected δ (log scale)")
        ax.set_title("Kalman δ Trajectory Across Folds")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "delta_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _run_universe_counts() -> None:
    from src.phase4_defense.orchestrator import (
        FOLD_SCHEDULE, load_phase1_pairs, _METRICS_DIR
    )

    rows = []
    for spec in FOLD_SCHEDULE:
        pairs_df = load_phase1_pairs(spec.fold_n)
        n_pairs = len(pairs_df) if pairs_df is not None else 0
        if pairs_df is not None and not pairs_df.empty:
            n_tickers = len(set(
                pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
            ))
        else:
            n_tickers = 0

        rows.append({
            "fold":          spec.fold_n,
            "trading_month": spec.trading_month,
            "form_start":    spec.form_start,
            "form_end":      spec.form_end,
            "n_surviving_pairs": n_pairs,
            "n_unique_tickers":  n_tickers,
        })

    result = pd.DataFrame(rows)
    result.to_csv(_METRICS_DIR / "universe_counts.csv", index=False)
    log.info("Universe counts saved (%d folds)", len(result))
    print(f"\n  Universe counts: median pairs/fold = {result['n_surviving_pairs'].median():.0f}, "
          f"max = {result['n_surviving_pairs'].max()}")


def _extract_nc_latency(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    # NC bootstrap (§7)
    nc_cols = ["fold", "trading_month", "nc_threshold", "nc_pass", "sharpe"]
    nc_df = fold_metrics[[c for c in nc_cols if c in fold_metrics.columns]].copy()
    nc_df.to_csv(_METRICS_DIR / "nc_bootstrap.csv", index=False)
    n_pass = nc_df["nc_pass"].sum() if "nc_pass" in nc_df.columns else 0
    print(f"\n  NC bootstrap: {n_pass}/{len(nc_df)} folds pass (Primary > NC threshold)")

    # Latency (§8)
    lat_cols = ["fold", "trading_month", "t1_sharpe", "t5_sharpe", "t10_sharpe", "latency_pass"]
    lat_df = fold_metrics[[c for c in lat_cols if c in fold_metrics.columns]].copy()
    lat_df.to_csv(_METRICS_DIR / "latency_decay.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for col, lag, color in [
            ("t1_sharpe", "t+1", "#2ca02c"),
            ("t5_sharpe", "t+5", "#ff7f0e"),
            ("t10_sharpe","t+10","#d62728"),
        ]:
            if col in lat_df.columns:
                ax.plot(range(len(lat_df)), lat_df[col], "o-",
                        markersize=3, linewidth=1.2, alpha=0.8,
                        label=f"Lag {lag} (mean={lat_df[col].mean():.2f})",
                        color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Sharpe")
        ax.set_title("Latency Sweep: Alpha Decay vs Execution Lag")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "latency_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _print_exit_summary(exit_df: pd.DataFrame) -> None:
    tot = exit_df["n_trades_parsed"].sum()
    eos = exit_df["n_eos"].sum()
    zc  = exit_df["n_zero_cross"].sum()
    sl  = exit_df.get("n_sl", pd.Series(0)).sum()

    print(f"\n  Exit reasons (aggregate across all folds):")
    print(f"    EOS exits:       {int(eos):6d}  ({eos/tot:.1%})")
    print(f"    Zero-cross:      {int(zc):6d}  ({zc/tot:.1%})")
    if sl > 0:
        print(f"    Stop-loss:       {int(sl):6d}  ({sl/tot:.1%})")
    print(f"    Avg net EOS:     {exit_df['avg_net_eos'].mean():.1f} bps")
    print(f"    Avg net ZC:      {exit_df['avg_net_zc'].mean():.1f} bps")


def _check_outputs() -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    required_metrics = [
        "sharpe_distribution.csv",   # §1
        "regime_sharpes.csv",         # §2
        "pair_persistence.csv",       # §3 (optional)
        "volume_strat.csv",           # §4 (optional)
        "overfitting_diagnostics.csv",# §5
        "oat_sensitivity.csv",        # §6
        "nc_bootstrap.csv",           # §7
        "latency_decay.csv",          # §8
        "exit_reasons.csv",           # §9
        "cost_decomp.csv",            # §10
        "delta_trajectory.csv",       # §11
        "universe_counts.csv",        # §12
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
        path = _METRICS_DIR / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/metrics/{fname:<35} {status}")

    for fname in required_figures:
        path = _FIGURES_DIR / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/figures/{fname:<35} {status}")

    print()
    if all_ok:
        log.info("All whitepaper outputs present.")
    else:
        log.warning("Some outputs missing. Run with --persistence --volume-strat for §3/§4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Defense Analysis")
    parser.add_argument("--persistence",     action="store_true",
                        help="Run P_2022 pair persistence (~30 min)")
    parser.add_argument("--volume-strat",    action="store_true",
                        help="Run volume stratification (~15 min)")
    parser.add_argument("--structural-oat",  action="store_true",
                        help="Run structural OAT variations (~20 min)")
    parser.add_argument("--all",             action="store_true",
                        help="Run all analyses including slow ones")
    args = parser.parse_args()
    main(args)
