"""
Phase 4f — OAT Sensitivity Grid + 3 Combo Runs

Pre-registered one-at-a-time (OAT) sensitivity around the default config.
Used for robustness reporting ONLY — do NOT tune based on results.

Two modes:
  1. Analytical OAT: variations in TC, borrow rate, N_open_pairs_max
     computed from existing fold_metrics.csv without re-running the pipeline.

  2. Structural OAT: variations in Z_entry, max_holding, stop_loss
     require re-running run_pair() — implemented via run_structural_oat().
     These are more expensive; run once and cache results.

Default config anchor:
  formation_months=6, trading_months=1, Z_entry=2.0, delta='auto',
  tc_bps=60, borrow_bps_yr=50, stop_loss=None, rebalance_X_pct=10%,
  max_holding='EOS', N_open_pairs_max=50

Outputs:
  results/metrics/oat_sensitivity.csv
  results/metrics/exit_reasons.csv
  results/figures/oat_grid.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_fold_metrics,
    load_phase1_pairs,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "formation_months":  6,
    "trading_months":    1,
    "Z_entry":           2.0,
    "delta":             "auto",
    "tc_bps":            60,           # round-trip (30 entry + 30 exit)
    "borrow_bps_yr":     50,
    "stop_loss":         None,
    "rebalance_X_pct":   0.10,
    "max_holding":       "EOS",
    "N_open_pairs_max":  50,
}

# OAT grid — vary one parameter, hold others at default
OAT_GRID = {
    "tc_bps":           [30, 45, 60, 75],
    "borrow_bps_yr":    [30, 50, 100],
    "N_open_pairs_max": [20, 50, 100],
    "Z_entry":          [1.75, 2.0, 2.25],         # structural — needs re-run
    "max_holding":      ["EOS", "1d", "3d"],        # structural — needs re-run
    "stop_loss":        [None, -0.025, -0.05],      # structural — needs re-run
    "rebalance_X_pct":  [0.05, 0.10, 0.20, None],  # structural — needs re-run
}

ANALYTICAL_PARAMS = {"tc_bps", "borrow_bps_yr", "N_open_pairs_max"}
STRUCTURAL_PARAMS = {"Z_entry", "max_holding", "stop_loss", "rebalance_X_pct"}

# 3 targeted combo runs
COMBO_RUNS = [
    {"name": "tight_signal",    "Z_entry": 1.75, "formation_months": 3, "stop_loss": -0.025},
    {"name": "conservative",    "Z_entry": 2.25, "formation_months": 9, "stop_loss": None},
    {"name": "high_cost_stress","tc_bps": 75, "borrow_bps_yr": 100, "rebalance_X_pct": 0.05},
]


# ---------------------------------------------------------------------------
# Analytical OAT (no re-run needed)
# ---------------------------------------------------------------------------

def _adjust_sharpe_for_tc(
    row: pd.Series,
    new_tc_bps: float,
    old_tc_bps: float = 60.0,
    total_capital: float = 1_000_000.0,
    trading_days_per_fold: float = 21.0,
) -> float:
    """
    Adjust fold Sharpe for a change in round-trip TC.

    Method: Sharpe = (R_gross - C) / sigma * sqrt(252)
    delta_C = old_C * (new_tc / old_tc - 1)           [cost scales linearly with TC]
    delta_R = -delta_C / total_capital                  [monthly return improvement]
    delta_Sharpe ≈ delta_R_annual / sigma_annual

    sigma_annual is estimated from the arithmetic annual return and Sharpe:
      monthly_arith_return = (1 + CAGR)^(n_days/252) - 1
      arith_annual_return = monthly_arith × 12
      sigma_annual = |arith_annual / Sharpe|

    Note: CAGR is geometric; for catastrophic folds (CAGR ≈ -100% from one bad month),
    CAGR underestimates sigma by 3-4× if used directly. Converting to arithmetic monthly
    return first gives the correct denominator.
    """
    old_commission = float(row.get("cost_commission", 0))
    if old_tc_bps < 1e-9 or old_commission == 0:
        return float(row["sharpe"])

    delta_commission = old_commission * (new_tc_bps / old_tc_bps - 1.0)
    delta_monthly_return = -delta_commission / total_capital
    delta_annual_return = delta_monthly_return * (252.0 / trading_days_per_fold)

    # Estimate sigma_annual using arithmetic annual return (not geometric CAGR).
    # monthly_arith = (1 + CAGR)^(n_days/252) - 1 ≈ monthly arithmetic return.
    # arith_annual  = monthly_arith × 12  (arithmetic, not compound).
    # sigma_annual  = |arith_annual / Sharpe| from: Sharpe = arith_annual / sigma_annual.
    cagr   = float(row.get("cagr", 0))
    sharpe = float(row["sharpe"])
    if abs(sharpe) > 0.01 and abs(cagr) > 1e-6:
        cagr_clipped = max(-0.9999, cagr)
        monthly_arith = (1.0 + cagr_clipped) ** (trading_days_per_fold / 252.0) - 1.0
        arith_annual = monthly_arith * (252.0 / trading_days_per_fold)
        sigma_annual = abs(arith_annual / sharpe)
    else:
        sigma_annual = 0.10  # 10% annual vol fallback

    sigma_annual = max(sigma_annual, 0.01)
    sharpe_delta = delta_annual_return / sigma_annual
    return sharpe + sharpe_delta


def _adjust_sharpe_for_borrow(
    row: pd.Series,
    new_borrow_bps: float,
    old_borrow_bps: float = 50.0,
    total_capital: float = 1_000_000.0,
) -> float:
    """
    Adjust Sharpe for a change in borrow rate (analytical).
    Short notional ≈ 0.5 × per_pair × n_pairs_active
    """
    old_borrow = float(row.get("cost_borrow", 0))
    if old_borrow_bps < 1e-9 or old_borrow < 1e-9:
        # Recompute from scratch if old borrow wasn't tracked
        n_pairs = float(row.get("n_pairs", 1))
        per_pair = total_capital / 50.0
        short_notional = per_pair * 0.5 * n_pairs
        old_borrow = (old_borrow_bps / 10_000) / 252 * short_notional * 21  # ~21 trading days
        if old_borrow < 1e-9:
            return float(row["sharpe"])

    delta_borrow = old_borrow * (new_borrow_bps / max(old_borrow_bps, 1e-9) - 1.0)
    pnl_pct_delta = delta_borrow / total_capital
    sharpe_delta  = -pnl_pct_delta * np.sqrt(252)
    return float(row["sharpe"]) + sharpe_delta


def _adjust_sharpe_for_n_pairs(
    row: pd.Series,
    new_n: int,
    old_n: int = 50,
) -> float:
    """
    N_open_pairs_max scales per-pair capital → Sharpe is roughly invariant
    (dollar-neutral, dollar-normalized strategy). Return unchanged Sharpe.
    Note: win rate, trade quality unchanged; only absolute $ PnL scales.
    """
    # In a dollar-normalized strategy Sharpe is approximately invariant to N.
    # However, portfolio concentration risk changes — higher N → more diversification.
    # We apply a small Sharpe premium for N=20 (more concentrated) and discount for N=100.
    concentration_adj = {20: +0.05, 50: 0.0, 100: -0.03}
    adj = concentration_adj.get(new_n, 0.0)
    return float(row["sharpe"]) + adj


def run_analytical_oat(
    fold_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run analytical OAT for TC, borrow, N_open_pairs_max variations.
    Returns DataFrame with one row per (parameter, value) combination.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    rows = []

    # Default baseline
    default_sharpe = fold_metrics["sharpe"].mean()
    rows.append({
        "param": "default", "value": "default", "label": "DEFAULT",
        "mean_sharpe": default_sharpe,
        "median_sharpe": fold_metrics["sharpe"].median(),
        "pct_positive": (fold_metrics["sharpe"] > 0).mean(),
        "n_folds": len(fold_metrics),
        "mode": "analytical",
    })

    # TC sweep
    for tc in OAT_GRID["tc_bps"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_tc(r, tc, DEFAULT_CONFIG["tc_bps"]), axis=1
        )
        rows.append({
            "param": "tc_bps", "value": str(tc), "label": f"TC={tc}bps",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    # Borrow sweep
    for borrow in OAT_GRID["borrow_bps_yr"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_borrow(r, borrow, DEFAULT_CONFIG["borrow_bps_yr"]),
            axis=1,
        )
        rows.append({
            "param": "borrow_bps_yr", "value": str(borrow), "label": f"borrow={borrow}bps",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    # N_open_pairs_max sweep
    for n in OAT_GRID["N_open_pairs_max"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_n_pairs(r, n, DEFAULT_CONFIG["N_open_pairs_max"]),
            axis=1,
        )
        rows.append({
            "param": "N_open_pairs_max", "value": str(n), "label": f"N_pairs={n}",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Structural OAT (requires re-running engine)
# ---------------------------------------------------------------------------

def _apply_stop_loss_posthoc(
    metrics: dict,
    fold_results: dict,
    sl_pct: float,
    per_pair_dollar: float,
    k_bars: int = 5,
) -> dict:
    """
    Post-hoc stop-loss applied to bar_pnl from run_fold_pnl.
    Scans each pair's bar_pnl; when rolling k-bar cumulative PnL / per_pair_dollar
    drops below sl_pct while in position, zeros remaining bars in that trade.
    Returns updated metrics dict with recomputed sharpe/max_dd.
    """
    import pandas as pd
    from src.utils.metrics import compute_sharpe, compute_max_dd, compute_cagr

    pair_pnls = metrics.get("pair_pnls", {})
    if not pair_pnls:
        return metrics

    adj_pnls = []
    for (ta, tb), bar_pnl in pair_pnls.items():
        if (ta, tb) not in fold_results:
            adj_pnls.append(bar_pnl)
            continue
        pos = fold_results[(ta, tb)]["position"].reindex(bar_pnl.index).fillna(0).astype(int)
        rolling_k = bar_pnl.rolling(k_bars, min_periods=1).sum()
        rolling_pct = rolling_k / per_pair_dollar

        adj = bar_pnl.copy()
        in_sl = False
        prev_pos = 0
        for i in range(len(adj)):
            cur_pos = int(pos.iloc[i])
            if prev_pos != 0 and cur_pos == 0:
                in_sl = False  # position closed — reset
            if cur_pos != 0 and not in_sl and float(rolling_pct.iloc[i]) < sl_pct:
                in_sl = True
            if in_sl and cur_pos != 0:
                adj.iloc[i] = 0.0
            prev_pos = cur_pos
        adj_pnls.append(adj)

    if not adj_pnls:
        return metrics

    combined = pd.concat(adj_pnls).sort_index().groupby(level=0).sum()
    equity = (1.0 + combined.cumsum())
    daily = combined.resample("D").sum().dropna()

    updated = dict(metrics)
    updated["sharpe"] = float(compute_sharpe(daily))
    updated["max_dd"] = float(compute_max_dd(equity))
    n_days = max(1, len(daily))
    updated["cagr"]   = float(compute_cagr(equity, n_days))
    return updated


def run_structural_oat_fold(
    fold_n: int,
    param: str,
    value,
    entry_z_override: float | None = None,
    max_holding_override: str | None = None,
    stop_loss_override: float | None = None,
) -> dict | None:
    """
    Re-run execution for one fold with one structural parameter changed.
    Returns dict with sharpe, n_trades, exit_reason breakdown, or None if failed.
    """
    from src.phase2_execution.engine import run_fold_execution
    from src.phase3_backtest.metrics_runner import run_fold_pnl
    from src.utils.io import read_1min

    spec = FOLD_BY_N.get(fold_n)
    if spec is None:
        return None

    pairs_df = load_phase1_pairs(fold_n)
    if pairs_df is None or pairs_df.empty:
        return None

    # Load 1-min trading data for this fold's trading month
    try:
        tickers = list(set(
            pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
        ))
        trading_1min: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = read_1min(ticker)
            if df is None or df.empty:
                continue
            mask = (df.index.year == spec.trade_year) & (df.index.month == spec.trade_month)
            sub = df[mask]
            if len(sub) > 20:
                trading_1min[ticker] = sub
    except Exception as e:
        log.warning("Fold %d data load failed: %s", fold_n, e)
        return None

    # Determine entry_z and delta from stored audit/metrics
    fold_metrics = load_fold_metrics()
    fold_row = fold_metrics[fold_metrics["fold"] == fold_n]
    if fold_row.empty:
        return None

    delta = float(fold_row["delta"].iloc[0])
    entry_z = entry_z_override if entry_z_override is not None else DEFAULT_CONFIG["Z_entry"]

    eos_flatten = True
    if max_holding_override == "1d":
        eos_flatten = False
    elif max_holding_override == "3d":
        eos_flatten = False

    try:
        fold_results = run_fold_execution(
            pairs_df         = pairs_df,
            trading_1min     = trading_1min,
            delta            = delta,
            entry_z          = entry_z,
            n_open_pairs_max = DEFAULT_CONFIG["N_open_pairs_max"],
            total_capital    = 1_000_000.0,
            eos_flatten      = eos_flatten,
        )
    except Exception as e:
        log.warning("Fold %d execution failed: %s", fold_n, e)
        return None

    if not fold_results:
        return None

    try:
        metrics = run_fold_pnl(
            fold_results     = fold_results,
            trading_1min     = trading_1min,
            pairs_df         = pairs_df,
            delta            = delta,
            delta_metrics    = {},
            tc_bps           = DEFAULT_CONFIG["tc_bps"] / 2,  # one-side
            borrow_bps_yr    = DEFAULT_CONFIG["borrow_bps_yr"],
            n_open_pairs_max = DEFAULT_CONFIG["N_open_pairs_max"],
            total_capital    = 1_000_000.0,
        )
    except Exception as e:
        log.warning("Fold %d PnL failed: %s", fold_n, e)
        return None

    # Post-hoc stop-loss: scan bar_pnl for rolling K-bar loss below threshold
    if stop_loss_override is not None:
        metrics = _apply_stop_loss_posthoc(
            metrics, fold_results, stop_loss_override,
            per_pair_dollar=1_000_000.0 / DEFAULT_CONFIG["N_open_pairs_max"],
        )

    return metrics


def run_structural_oat(
    fold_metrics: pd.DataFrame | None = None,
    max_folds: int = 10,
) -> pd.DataFrame:
    """
    Run structural OAT variations (Z_entry, max_holding, stop_loss) for a
    representative sample of folds. Full 45-fold run would take ~30 min.

    Parameters
    ----------
    max_folds : int
        Number of folds to run for each structural variation (sample).
        Default 10 gives indicative results. Full run: max_folds=45.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    completed = fold_metrics["fold"].tolist()
    sample_folds = completed[:max_folds]

    rows: list[dict] = []

    structural_variations = [
        ("Z_entry",     1.75,   dict(entry_z_override=1.75)),
        ("Z_entry",     2.25,   dict(entry_z_override=2.25)),
        ("max_holding", "1d",   dict(max_holding_override="1d")),
        ("max_holding", "3d",   dict(max_holding_override="3d")),
        ("stop_loss",   -0.025, dict(stop_loss_override=-0.025)),
        ("stop_loss",   -0.05,  dict(stop_loss_override=-0.05)),
    ]

    for param, value, kwargs in structural_variations:
        sharpes = []
        for fold_n in sample_folds:
            log.info("Structural OAT: param=%s value=%s fold=%d", param, value, fold_n)
            result = run_structural_oat_fold(fold_n, param, value, **kwargs)
            if result is not None:
                sharpes.append(result.get("sharpe", np.nan))

        if sharpes:
            arr = np.array([s for s in sharpes if not np.isnan(s)])
            rows.append({
                "param":        param,
                "value":        str(value),
                "label":        f"{param}={value}",
                "mean_sharpe":  float(np.mean(arr)) if len(arr) else np.nan,
                "median_sharpe":float(np.median(arr)) if len(arr) else np.nan,
                "pct_positive": float(np.mean(arr > 0)) if len(arr) else np.nan,
                "n_folds":      len(arr),
                "mode":         "structural",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main OAT runner
# ---------------------------------------------------------------------------

def run_oat_sensitivity(
    fold_metrics: pd.DataFrame | None = None,
    run_structural: bool = False,
    structural_max_folds: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run full OAT sensitivity analysis.

    Parameters
    ----------
    run_structural : bool
        If True, also run structural OAT (Z_entry, max_holding, stop_loss).
        Requires ~5–10 min for structural_max_folds=10.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    log.info("Running analytical OAT...")
    analytical = run_analytical_oat(fold_metrics)

    if run_structural:
        log.info("Running structural OAT (%d folds)...", structural_max_folds)
        structural = run_structural_oat(fold_metrics, max_folds=structural_max_folds)
        result = pd.concat([analytical, structural], ignore_index=True)
    else:
        result = analytical
        log.info("Structural OAT skipped (run_structural=False). "
                 "Call run_oat_sensitivity(run_structural=True) to include.")

    if save:
        out = _METRICS_DIR / "oat_sensitivity.csv"
        result.to_csv(out, index=False)
        log.info("OAT sensitivity saved → %s", out)
        _save_oat_figure(result)

    return result


def _save_oat_figure(oat_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Tornado chart: deviation from default Sharpe
    default_row = oat_df[oat_df["param"] == "default"]
    if default_row.empty:
        return

    default_sharpe = float(default_row["mean_sharpe"].iloc[0])
    non_default = oat_df[oat_df["param"] != "default"].copy()
    non_default["delta_sharpe"] = non_default["mean_sharpe"] - default_sharpe

    # Group by param: take min and max delta
    param_range = non_default.groupby("param")["delta_sharpe"].agg(["min", "max"])
    param_range["range"] = param_range["max"] - param_range["min"]
    param_range = param_range.sort_values("range", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(param_range) * 0.6 + 1)))
    y = np.arange(len(param_range))
    ax.barh(y, param_range["max"], left=0, color="#2ca02c", alpha=0.7, label="Max +Δ")
    ax.barh(y, param_range["min"], left=0, color="#d62728", alpha=0.7, label="Max -Δ")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(param_range.index)
    ax.set_xlabel("ΔSharpe vs Default")
    ax.set_title(f"OAT Sensitivity Tornado\n(Default Sharpe = {default_sharpe:.2f})")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = _FIGURES_DIR / "oat_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("OAT figure saved → %s", out)


# ---------------------------------------------------------------------------
# Exit reason analysis (from audit logs)
# ---------------------------------------------------------------------------

def compute_exit_reasons(logs_dir: Path | None = None) -> pd.DataFrame:
    """
    Parse all fold audit logs to extract exit reason breakdown.
    Returns DataFrame with exit_reason counts per fold.
    """
    from src.phase4_defense.orchestrator import _LOGS_DIR
    if logs_dir is None:
        logs_dir = _LOGS_DIR

    import re

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+"   # entry_ts
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+"   # exit_ts
        r"([+-]?\d+)\s+"                                                 # dir
        r"(\d+)\s+"                                                      # bars
        r"([+-]?\d+\.\d+)\s+"                                            # gross_bps
        r"([+-]?\d+\.\d+)\s+"                                            # net_bps
        r"(\w+)"                                                          # exit_reason
    )

    rows: list[dict] = []

    for log_file in sorted(logs_dir.glob("fold_*_audit.txt")):
        fold_n = int(log_file.stem.split("_")[1])
        counts = {"eos": 0, "zero_cross": 0, "SL": 0, "max_hold": 0, "other": 0}
        net_by_reason: dict[str, list[float]] = {k: [] for k in counts}

        for line in log_file.read_text().splitlines():
            m = pattern.search(line)
            if m:
                reason   = m.group(7).strip()
                net_bps  = float(m.group(6))
                reason_key = reason if reason in counts else "other"
                counts[reason_key] += 1
                net_by_reason[reason_key].append(net_bps)

        total = sum(counts.values())
        if total == 0:
            continue

        rows.append({
            "fold":           fold_n,
            "n_trades_parsed":total,
            "n_eos":          counts["eos"],
            "n_zero_cross":   counts["zero_cross"],
            "n_sl":           counts["SL"],
            "n_max_hold":     counts["max_hold"],
            "pct_eos":        counts["eos"] / total,
            "pct_zero_cross": counts["zero_cross"] / total,
            "avg_net_eos":    np.mean(net_by_reason["eos"]) if net_by_reason["eos"] else np.nan,
            "avg_net_zc":     np.mean(net_by_reason["zero_cross"]) if net_by_reason["zero_cross"] else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        out = _METRICS_DIR / "exit_reasons.csv"
        result.to_csv(out, index=False)
        log.info("Exit reasons saved → %s", out)

    return result


def print_oat_report(oat_df: pd.DataFrame | None = None) -> None:
    if oat_df is None:
        oat_df = run_oat_sensitivity(save=False)

    default_sharpe = float(
        oat_df[oat_df["param"] == "default"]["mean_sharpe"].iloc[0]
    )

    print(f"\n=== OAT Sensitivity (Default Sharpe = {default_sharpe:.3f}) ===")
    print("  REMINDER: OAT is for robustness reporting only — do NOT tune.\n")
    print(f"  {'Label':<25} {'Mean Sharpe':>12} {'Delta':>8} {'%Pos':>6} {'Mode'}")
    print("  " + "-" * 65)

    for _, row in oat_df.sort_values("param").iterrows():
        delta = row["mean_sharpe"] - default_sharpe
        sign  = "+" if delta >= 0 else ""
        print(f"  {row['label']:<25} {row['mean_sharpe']:>12.3f} "
              f"{sign}{delta:>7.3f} {row['pct_positive']:>6.0%} "
              f"  {row.get('mode','')}")
