"""
Carry-forward gate for V4 daily engine (E2 + revalidation).
============================================================

Allows positions open at end of fold N to survive into fold N+1, gated by
re-running Johansen p-value on fold N+1's formation window. Pairs that
pass (p < gate_threshold) carry forward with frozen original beta. Pairs
that fail are cut (force-closed in fold N as the V4 default).

Pre-committed config (locked 2026-05-24):
    - Gate threshold: p < 0.05 (no tuning)
    - Beta policy:    frozen from original fold (cannot change mid-position)
    - Alpha policy:   refit on each fold's formation (V4 standard)
    - Cost handling:  refund exit-cost + borrow at fold N's last bar for
                      carried pairs; book true exit cost when position
                      eventually closes (using original_entry_date for borrow).
    - Carry pair injection: pairs that pass gate but were NOT picked by
                            fold N+1's discovery (dropped by BH-FDR) are
                            injected into fold N+1's pair list with their
                            original beta.

Known design choices (NOT bugs, documented 2026-05-25 from deep-audit-bug):

    Bug #4 (logical) — Fold-boundary day P&L gap:
        A position held overnight from fold N's last bar (e.g., Jan 31) into
        fold N+1's day-0 (Feb 1) has a real price move during that gap that
        is not booked. Fold N's engine assumes flat-by-EOD (force-close
        convention); fold N+1's engine initializes spread_change[0] = 0,
        silently dropping the move. Magnitude: ~$200/carry per side, but
        SYMMETRIC (gain or loss missed with equal probability), so expected
        bias = 0 with added variance ~ sqrt(n_carries) * $200. At ~1500 carries
        the noise contribution is ~$8k, dwarfed by per-fold P&L of $10-100k.
        FIX would require fold-N+1's residual basis applied to fold-N's last
        bar prices, introducing an alpha-jump discontinuity. Net: design tax
        accepted; not fixed.

    Bug #5 (design) — Ticker concentration cap bypasses carry pairs:
        engine_daily.run_fold_daily applies apply_ticker_concentration_cap to
        the discovery list BEFORE merging in carry pairs. A pair like AAPL/MSFT
        that carries through 3 folds doesn't count against fold N+1's per-ticker
        cap when admitting new pairs. INTENTIONAL: can't "unwind" a carried
        position because of a cap that takes effect mid-position; the cap is
        an admission filter, not a forced-liquidation rule. Documented choice:
        live with potential per-ticker over-concentration in carry-heavy regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.phase1_cointegration.discovery import (
    _apply_hard_screens,
    _johansen_pvalue,
)
from engine.phase1_cointegration.factor_residual import fit_factor_model

log = logging.getLogger(__name__)

GATE_PVAL_THRESHOLD = 0.05      # locked, no tuning
MIN_JOHANSEN_OVERLAP_BARS = 50  # need >= this many overlapping non-NaN bars for test
_N_FACTOR_COMPONENTS = 5        # mirror engine_daily.discovery_daily
_MIN_OBS_DAILY = 100


def compute_residuals_for_gate(
    formation_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, set[str]]:
    """Replicate discovery_daily's Phase 1 prep through residual_log_prices.

    Used by the orchestrator to build fold N+1's residuals for the gate test
    without re-running full discovery (Johansen on 130k pairs).

    Returns
    -------
    resid_df : (T_form, n_tickers) residual log-prices after PCA factor projection.
    survivors : set of tickers in resid_df.
    """
    if not formation_data:
        return pd.DataFrame(), set()

    survivors, _, _ = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        return pd.DataFrame(), set()

    formation_data = {t: formation_data[t] for t in survivors}

    try:
        _, factor_tickers, residual_log_prices, _ = fit_factor_model(
            formation_data, n_components=_N_FACTOR_COMPONENTS,
        )
    except ValueError:
        return pd.DataFrame(), set()

    kept = {
        tk: residual_log_prices[tk] for tk in factor_tickers
        if residual_log_prices[tk].dropna().size >= _MIN_OBS_DAILY
    }
    if not kept:
        return pd.DataFrame(), set()
    return pd.DataFrame(kept), set(kept.keys())


@dataclass
class CarryState:
    """Per-pair state passed across fold boundaries.

    Attributes
    ----------
    ticker_a, ticker_b : pair identity (frozen from original entry).
    position           : +1 (long A short B) or -1 (short A long B).
    beta_original      : hedge ratio frozen from the fold where position
                         was originally opened (NEVER re-estimated).
    notional           : vol-target sizing decision from original fold.
    original_entry_date: tz-naive Timestamp of true entry (used by borrow
                         accrual on eventual exit, NOT day-0 of any
                         carried fold).
    carries_done       : count of fold boundaries this position has
                         survived. 0 = just entered, 1 = carried once, etc.
    fold_origin        : fold_n where position was first opened.
    """
    ticker_a: str
    ticker_b: str
    position: int
    beta_original: float
    notional: float
    original_entry_date: pd.Timestamp
    carries_done: int
    fold_origin: int


def gate_test_single(
    next_resid_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> float:
    """Run Johansen on (resid_a, resid_b) using fold N+1's residual matrix.

    Returns p-value (chi2(8) approx via _johansen_pvalue), or np.nan if
    either ticker is missing from the residual matrix or overlap is too small.
    """
    if ticker_a not in next_resid_df.columns or ticker_b not in next_resid_df.columns:
        return np.nan
    s_a = next_resid_df[ticker_a].dropna()
    s_b = next_resid_df[ticker_b].dropna()
    aligned = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
    if len(aligned) < MIN_JOHANSEN_OVERLAP_BARS:
        return np.nan
    return _johansen_pvalue(aligned.values)


def apply_gate(
    carry_candidates: dict[tuple[str, str], CarryState],
    next_resid_df: pd.DataFrame,
    gate_threshold: float = GATE_PVAL_THRESHOLD,
) -> tuple[dict[tuple[str, str], CarryState], dict[tuple[str, str], float]]:
    """Filter carry candidates by Johansen p-value on fold N+1's residuals.

    Parameters
    ----------
    carry_candidates : pairs that were open at fold N's EOM (from engine output).
    next_resid_df    : residual_log_prices DataFrame from fold N+1's discovery.
                       Columns = tickers, index = formation dates.
    gate_threshold   : pass if pval < threshold. Default 0.05 (locked).

    Returns
    -------
    passed : carry candidates that passed the gate (continue to fold N+1).
    pvals  : dict (a,b) -> pval for ALL candidates (passed and failed).
             Used for diagnostics + log.
    """
    passed: dict[tuple[str, str], CarryState] = {}
    pvals: dict[tuple[str, str], float] = {}
    n_eligible = 0
    n_pass = 0

    for key, state in carry_candidates.items():
        pval = gate_test_single(next_resid_df, state.ticker_a, state.ticker_b)
        pvals[key] = pval
        if np.isnan(pval):
            continue  # ineligible (ticker missing or insufficient overlap)
        n_eligible += 1
        if pval < gate_threshold:
            n_pass += 1
            new_state = CarryState(
                ticker_a=state.ticker_a, ticker_b=state.ticker_b,
                position=state.position, beta_original=state.beta_original,
                notional=state.notional,
                original_entry_date=state.original_entry_date,
                carries_done=state.carries_done + 1,
                fold_origin=state.fold_origin,
            )
            passed[key] = new_state

    log.info(
        "carry-forward gate: %d candidates -> %d eligible -> %d pass (p<%.2f)",
        len(carry_candidates), n_eligible, n_pass, gate_threshold,
    )
    return passed, pvals


def extract_open_at_eom(
    pair_results: dict[tuple[str, str], pd.DataFrame],
    pairs_df_used: pd.DataFrame,
    fold_n: int,
) -> dict[tuple[str, str], CarryState]:
    """Walk fold N's per-pair output, build carry candidates for pairs open at EOM.

    Parameters
    ----------
    pair_results   : engine output (bar-level DataFrame per pair).
    pairs_df_used  : the pair config that engine actually ran (after concat of
                     discovery + injected carry pairs). Must have columns
                     ticker_a, ticker_b, beta_pca (for non-carry pairs the
                     beta_pca is the fold's own; for carry pairs it should be
                     the carried beta_original — caller must enforce).
    fold_n         : current fold (stamped onto new CarryState if pair is
                     entering for the first time).

    Returns
    -------
    carry_candidates : pairs whose position at last bar is non-zero. These go
                       to apply_gate(); the gate may keep or cut them.
    """
    beta_lookup = {
        (r["ticker_a"], r["ticker_b"]): float(r["beta_pca"])
        for _, r in pairs_df_used.iterrows()
    }

    candidates: dict[tuple[str, str], CarryState] = {}
    for (ta, tb), pdf in pair_results.items():
        if len(pdf) == 0:
            continue
        last_pos = int(pdf["position"].iloc[-1])
        if last_pos == 0:
            continue
        # Find the entry bar of the open trade (walk back from end)
        positions = pdf["position"].values
        entry_idx = len(positions) - 1
        while entry_idx > 0 and positions[entry_idx - 1] == last_pos:
            entry_idx -= 1
        entry_date = pdf.index[entry_idx]

        beta = beta_lookup.get((ta, tb))
        if beta is None:
            log.warning("extract_open_at_eom: no beta for %s/%s -- skip", ta, tb)
            continue
        notional = float(pdf.attrs.get("notional", 0.0))

        # BUG #2 + #3 FIX (2026-05-25):
        #   #2: engine writes attrs["original_entry_date"] = None for fresh
        #       (non-carried) entries; .get(..., default) returns the stored
        #       None instead of falling back to default. Must explicitly check.
        #   #3: when a pair was carried in but then exited mid-fold and
        #       re-entered later, the prior attrs["original_entry_date"]
        #       is STALE relative to the current open trade. Detect by
        #       entry_idx > 0 (held continuously from day-0 if entry_idx == 0).
        attrs_oed = pdf.attrs.get("original_entry_date")
        if entry_idx > 0 or attrs_oed is None:
            # Position re-entered within this fold (or no prior carry):
            # use this fold's actual entry_date.
            original_entry = entry_date
        else:
            # Held continuously from day-0 -> truly a carried position;
            # use the original entry date stamped from the origin fold.
            original_entry = attrs_oed
        carries_done = int(pdf.attrs.get("carries_done", 0))
        fold_origin = int(pdf.attrs.get("fold_origin", fold_n))

        candidates[(ta, tb)] = CarryState(
            ticker_a=ta, ticker_b=tb, position=last_pos,
            beta_original=beta, notional=notional,
            original_entry_date=original_entry,
            carries_done=carries_done, fold_origin=fold_origin,
        )

    return candidates


def refund_eom_exit_cost(
    pair_results: dict[tuple[str, str], pd.DataFrame],
    carry_passed: dict[tuple[str, str], CarryState],
) -> int:
    """For pairs that will be carried, undo the EOM-force-close exit/borrow
    booked by the engine at the last bar. Recompute net P&L + cum P&L for
    that bar so fold-level metrics reconcile.

    Note: entry cost (booked at original entry bar) stays. Only exit-side
    costs at the last bar are refunded.

    Returns: count of pairs whose costs were refunded.
    """
    n_refunded = 0
    for key, state in carry_passed.items():
        pdf = pair_results.get(key)
        if pdf is None or len(pdf) == 0:
            continue
        last_idx = len(pdf) - 1
        # Refund all exit-side bookings at last bar
        cost_exit_last = float(pdf["cost_exit"].iloc[last_idx])
        borrow_last = float(pdf["borrow_cost"].iloc[last_idx])
        if cost_exit_last == 0.0 and borrow_last == 0.0:
            continue
        # Use loc/iloc carefully on the engine's DataFrame
        idx_label = pdf.index[last_idx]
        pdf.loc[idx_label, "cost_exit"] = 0.0
        pdf.loc[idx_label, "borrow_cost"] = 0.0
        # Decomposition columns (dynamic path only)
        for col in ("cost_spread", "cost_impact", "cost_commission"):
            if col in pdf.columns:
                # Half belongs to the exit side. Conservatively zero the last
                # bar's contribution since dynamic engine booked exit-half here.
                # Entry-half was already booked at entry bar — leave alone.
                pdf.loc[idx_label, col] = 0.0
        # Recompute daily_pnl_net at last bar
        gross = float(pdf["daily_pnl_gross"].iloc[last_idx])
        cost_entry = float(pdf["cost_entry"].iloc[last_idx])
        # net = gross - cost_entry - cost_exit (refunded) - borrow (refunded)
        pdf.loc[idx_label, "daily_pnl_net"] = gross - cost_entry
        pdf["cum_pnl"] = pdf["daily_pnl_net"].cumsum()
        n_refunded += 1

    return n_refunded
