"""
Phase 4c — Pair Persistence Decay Curve

For P_2022 (pairs from Fold 1 formation Jul–Dec 2022 → trading Jan 2023,
which is Fold 7 in our schedule where formation = 2022-07-01 to 2022-12-31),
re-tests Johansen cointegration for each subsequent fold's formation window.

Output:
  results/metrics/pair_persistence.csv   — (fold, trading_month, n_pairs, n_still_passing, pct_passing)
  results/figures/persistence_line.png
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from engine.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_phase1_pairs,
    _METRICS_DIR,
    _FIGURES_DIR,
)
from engine.utils.io import read_5min

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# P_2022 target: pairs from formation Jul–Dec 2022 (Fold 7).
# Fold 7 may have zero surviving pairs (the 2022 H2 window was a stressed period
# with few cointegrated pairs passing BH-FDR). If empty, we fall back to the
# nearest fold with actual pairs — Fold 6 (formation Jun–Nov 2022).
_P2022_FOLD_PREFERRED = 7
_P2022_FOLD_FALLBACK  = 6   # formation Jun-Nov 2022; confirmed 64 pairs
_JOHANSEN_PVAL_THRESH = 0.05


def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Run Johansen trace test on pair, return min p-value (trace rank 0)."""
    mat = np.column_stack([log_a, log_b])
    if len(mat) < 10:
        return 1.0
    try:
        res = coint_johansen(mat, det_order=0, k_ar_diff=1)
        # p-values not directly returned; use critical values at 5% for trace stat
        # Trace stat > cv_5pct → reject H0: no cointegration
        trace_stat = res.lr1[0]      # rank 0 trace statistic
        cv_5pct    = res.cvt[0, 1]  # 5% critical value for rank 0
        # Return approximate p: <0.05 if stat > cv_5pct
        return 0.01 if trace_stat > cv_5pct else 0.20
    except Exception:
        return 1.0


def _load_formation_5min(
    fold_n: int,
    tickers: list[str],
) -> dict[str, pd.Series]:
    """Load 5-min log_close for given tickers over fold N's formation window."""
    spec = FOLD_BY_N[fold_n]
    cache: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = read_5min(ticker)
            if df is None or df.empty:
                continue
            mask = (
                (df.index >= spec.form_start) &
                (df.index <= spec.form_end)
            )
            sub = df.loc[mask, "log_close"].dropna()
            if len(sub) >= 10:
                cache[ticker] = sub
        except Exception:
            pass
    return cache


def run_persistence(
    save: bool = True,
) -> pd.DataFrame:
    """
    Re-test Johansen on P_2022 pairs for each fold 7–45.

    Returns DataFrame with columns:
      [fold, trading_month, n_pairs_tested, n_still_passing, pct_passing]
    """
    # Load P_2022: try preferred fold (7 = Jul-Dec 2022 formation), fall back to fold 6.
    p2022_fold = _P2022_FOLD_PREFERRED
    p2022_df   = load_phase1_pairs(p2022_fold)
    if p2022_df is None or p2022_df.empty:
        log.warning(
            "Fold %d has no pairs (expected for stressed 2022 H2 window). "
            "Falling back to fold %d for P_2022.",
            _P2022_FOLD_PREFERRED, _P2022_FOLD_FALLBACK,
        )
        p2022_fold = _P2022_FOLD_FALLBACK
        p2022_df   = load_phase1_pairs(p2022_fold)

    if p2022_df is None or p2022_df.empty:
        log.error("P_2022 fallback fold %d also empty. Cannot run persistence.", p2022_fold)
        return pd.DataFrame()

    log.info("Using fold %d as P_2022 source (%d pairs)", p2022_fold, len(p2022_df))
    p2022_pairs = list(zip(p2022_df["ticker_a"], p2022_df["ticker_b"]))
    all_tickers  = list(set(p2022_df["ticker_a"].tolist() + p2022_df["ticker_b"].tolist()))
    log.info("P_2022: %d pairs, %d unique tickers", len(p2022_pairs), len(all_tickers))

    rows = []

    for spec in FOLD_SCHEDULE:
        if spec.fold_n < p2022_fold:
            continue  # only track from source fold onward

        log.info("Persistence check: Fold %d [%s → %s]",
                 spec.fold_n, spec.form_start, spec.form_end)

        series_cache = _load_formation_5min(spec.fold_n, all_tickers)

        n_tested  = 0
        n_passing = 0

        for (ta, tb) in p2022_pairs:
            if ta not in series_cache or tb not in series_cache:
                continue

            sa, sb = series_cache[ta].align(series_cache[tb], join="inner")
            valid = ~(sa.isna() | sb.isna())
            log_a = sa[valid].values
            log_b = sb[valid].values

            if len(log_a) < 20:
                continue

            n_tested += 1
            pval = _johansen_pval(log_a, log_b)
            if pval < _JOHANSEN_PVAL_THRESH:
                n_passing += 1

        pct = (n_passing / n_tested) if n_tested > 0 else np.nan
        rows.append({
            "fold":            spec.fold_n,
            "trading_month":   spec.trading_month,
            "n_pairs_tested":  n_tested,
            "n_still_passing": n_passing,
            "pct_passing":     pct,
        })
        log.info("  → %d/%d passing (%.1f%%)", n_passing, n_tested,
                 100 * pct if not np.isnan(pct) else 0)

    result = pd.DataFrame(rows)

    if save and not result.empty:
        out = _METRICS_DIR / "pair_persistence.csv"
        result.to_csv(out, index=False)
        log.info("Pair persistence saved → %s", out)
        _save_figure(result)

    return result


def _save_figure(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["trading_month"], df["pct_passing"] * 100,
            marker="o", linewidth=2, color="#1f77b4", markersize=4)
    ax.fill_between(range(len(df)), df["pct_passing"] * 100, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Trading Month")
    ax.set_ylabel("% of P_2022 Pairs Still Cointegrated")
    ax.set_title("P_2022 Pair Persistence: % Passing Johansen (5%) at Each Fold")
    ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
    ax.set_xticklabels(df["trading_month"].iloc[::max(1, len(df) // 10)],
                       rotation=45, ha="right", fontsize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = _FIGURES_DIR / "persistence_line.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Persistence figure saved → %s", out)
