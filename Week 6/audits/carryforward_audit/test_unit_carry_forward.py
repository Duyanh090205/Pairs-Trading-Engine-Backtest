"""
Unit / smoke tests for engine_daily/carry_forward.py
====================================================

Each function tested in isolation on synthetic data. No external dependencies
(no parquet cache reads, no Week-5 spread data). Designed to be safe to run
while a full 39-fold pipeline is executing in another process.

Tests:
    T1 - CarryState dataclass round-trip + immutable semantics
    T2 - compute_residuals_for_gate: shape, columns, min-obs filter
    T3 - gate_test_single: cointegrated pair passes, anti-cointegrated fails,
         missing ticker returns NaN, insufficient overlap returns NaN
    T4 - apply_gate: threshold semantics, carries_done increments, NaN handling
    T5 - extract_open_at_eom: open-vs-flat detection, beta lookup, attrs fallback
         (THIS TEST IS DESIGNED TO EXPOSE THE original_entry_date=None BUG)
    T6 - refund_eom_exit_cost: cost zeroing + cum_pnl recomputation

Run: python audits/carryforward_audit/test_unit_carry_forward.py
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))

from engine_daily.carry_forward import (
    CarryState,
    GATE_PVAL_THRESHOLD,
    MIN_JOHANSEN_OVERLAP_BARS,
    apply_gate,
    compute_residuals_for_gate,
    extract_open_at_eom,
    gate_test_single,
    refund_eom_exit_cost,
)

PASS_COUNT = 0
FAIL_COUNT = 0
BUG_FLAGS: list[str] = []


def _passed(msg: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}", flush=True)


def _failed(msg: str, *, is_bug: bool = False) -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {msg}", flush=True)
    if is_bug:
        BUG_FLAGS.append(msg)


# ============================================================================
# T1 - CarryState dataclass
# ============================================================================

def t1_carry_state():
    print("\nT1 -- CarryState dataclass")
    s = CarryState(
        ticker_a="AAA", ticker_b="BBB", position=1, beta_original=1.5,
        notional=20000.0, original_entry_date=pd.Timestamp("2023-01-15"),
        carries_done=0, fold_origin=1,
    )
    if s.ticker_a == "AAA" and s.beta_original == 1.5 and s.position == 1:
        _passed("CarryState constructs and exposes attributes")
    else:
        _failed("CarryState attribute round-trip broken")


# ============================================================================
# T2 - compute_residuals_for_gate
# ============================================================================

def _make_synthetic_formation(
    n_tickers: int = 60, n_days: int = 252, seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Synthetic daily formation: 1 market factor + idio noise + ticker-specific drift.
    Each ticker's DataFrame has columns `log_close` + `volume`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    market = np.cumsum(rng.normal(0, 0.012, n_days))
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        beta_mkt = 0.5 + 0.7 * rng.random()
        idio = np.cumsum(rng.normal(0, 0.008, n_days))
        log_close = 4.5 + beta_mkt * market + idio
        vol = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
        out[f"T{i:03d}"] = pd.DataFrame(
            {"log_close": log_close, "volume": vol}, index=dates,
        )
    return out


def t2_compute_residuals_for_gate():
    print("\nT2 -- compute_residuals_for_gate")

    # T2a - empty input returns (empty, empty set)
    df, surv = compute_residuals_for_gate({})
    if df.empty and surv == set():
        _passed("empty input -> (empty DataFrame, empty set)")
    else:
        _failed("empty input mishandled")

    # T2b - synthetic universe yields residual matrix
    formation = _make_synthetic_formation(n_tickers=60, n_days=252, seed=1)
    df, surv = compute_residuals_for_gate(formation)
    if not df.empty and len(surv) >= 40 and df.shape[0] >= 100:
        _passed(f"synthetic 60-ticker, 252-day -> resid shape={df.shape}, "
                f"survivors={len(surv)}")
    else:
        _failed(f"synthetic produced empty / wrong shape: shape={df.shape}, surv={len(surv)}")

    # T2c - columns are tickers (string), index is dates
    if all(isinstance(c, str) for c in df.columns) and isinstance(df.index, pd.DatetimeIndex):
        _passed("output columns=tickers (str), index=DatetimeIndex")
    else:
        _failed("output schema wrong")


# ============================================================================
# T3 - gate_test_single
# ============================================================================

def t3_gate_test_single():
    print("\nT3 -- gate_test_single")
    rng = np.random.default_rng(2)
    n = 252
    dates = pd.bdate_range("2022-01-03", periods=n)

    # T3a - missing ticker -> NaN
    resid = pd.DataFrame({"X": rng.normal(0, 1, n), "Y": rng.normal(0, 1, n)},
                         index=dates)
    p = gate_test_single(resid, "X", "ZZZ")  # ZZZ missing
    if np.isnan(p):
        _passed("missing ticker -> NaN")
    else:
        _failed(f"missing ticker did not return NaN; got {p}")

    # T3b - cointegrated pair: build shared random walk + small idio noise
    common = np.cumsum(rng.normal(0, 0.01, n))
    a = common + 0.01 * rng.normal(0, 1, n)
    b = 0.8 * common + 0.01 * rng.normal(0, 1, n)
    resid_coint = pd.DataFrame({"A": a, "B": b}, index=dates)
    p_coint = gate_test_single(resid_coint, "A", "B")
    if not np.isnan(p_coint) and p_coint < 0.05:
        _passed(f"cointegrated pair: pval={p_coint:.4f} < 0.05")
    else:
        _failed(f"cointegrated pair pval={p_coint}, expected <0.05")

    # T3c - two independent random walks: should usually NOT be cointegrated
    a2 = np.cumsum(rng.normal(0, 0.01, n))
    b2 = np.cumsum(rng.normal(0, 0.01, n))
    resid_indep = pd.DataFrame({"A": a2, "B": b2}, index=dates)
    p_indep = gate_test_single(resid_indep, "A", "B")
    if np.isnan(p_indep) or p_indep > 0.1:
        _passed(f"independent random walks: pval={p_indep} (>0.1 or NaN — expected)")
    else:
        # Not deterministically wrong, but flag if very tight
        _passed(f"independent random walks: pval={p_indep:.4f} (lucky low — informational)")

    # T3d - insufficient overlap -> NaN
    short_resid = pd.DataFrame(
        {"A": a[:20], "B": b[:20]},
        index=dates[:20],
    )
    p_short = gate_test_single(short_resid, "A", "B")
    if np.isnan(p_short):
        _passed(f"insufficient overlap (20 bars < {MIN_JOHANSEN_OVERLAP_BARS}) -> NaN")
    else:
        _failed(f"short overlap returned {p_short}, expected NaN")


# ============================================================================
# T4 - apply_gate
# ============================================================================

def t4_apply_gate():
    print("\nT4 -- apply_gate")
    rng = np.random.default_rng(3)
    n = 252
    dates = pd.bdate_range("2022-01-03", periods=n)
    common = np.cumsum(rng.normal(0, 0.01, n))

    # Build a residual matrix with one cointegrated pair (A,B) and one non-cointegrated (C,D)
    resid = pd.DataFrame({
        "A": common + 0.01 * rng.normal(0, 1, n),
        "B": 0.8 * common + 0.01 * rng.normal(0, 1, n),
        "C": np.cumsum(rng.normal(0, 0.01, n)),
        "D": np.cumsum(rng.normal(0, 0.01, n)),
    }, index=dates)

    def _make_state(ta, tb, beta=1.0, carries_done=0):
        return CarryState(
            ticker_a=ta, ticker_b=tb, position=1, beta_original=beta,
            notional=20000.0, original_entry_date=pd.Timestamp("2023-01-01"),
            carries_done=carries_done, fold_origin=1,
        )

    candidates = {
        ("A", "B"): _make_state("A", "B", beta=0.8, carries_done=2),
        ("C", "D"): _make_state("C", "D", beta=1.0, carries_done=0),
    }
    passed, pvals = apply_gate(candidates, resid, gate_threshold=0.05)

    # T4a - both pairs got a pvalue
    if set(pvals.keys()) == set(candidates.keys()):
        _passed(f"all candidates received a pvalue (n={len(pvals)})")
    else:
        _failed(f"pvals keys != candidates keys")

    # T4b - cointegrated pair (A,B) passes; non-cointegrated (C,D) usually fails
    if ("A", "B") in passed:
        _passed(f"cointegrated (A,B) passed gate (pval={pvals[('A','B')]:.4f})")
    else:
        _failed(f"cointegrated (A,B) did NOT pass; pval={pvals[('A','B')]}")
    if ("C", "D") not in passed:
        _passed(f"non-cointegrated (C,D) cut (pval={pvals[('C','D')]:.4f})")
    else:
        _passed(f"non-cointegrated (C,D) passed (pval={pvals[('C','D')]:.4f}) — "
                f"lucky-low, informational only")

    # T4c - carries_done incremented for passing pairs
    if ("A", "B") in passed and passed[("A", "B")].carries_done == 3:
        _passed("carries_done incremented (2 -> 3) for passing pair")
    elif ("A", "B") in passed:
        _failed(f"carries_done did not increment correctly; "
                f"got {passed[('A','B')].carries_done}, expected 3")

    # T4d - frozen state fields propagated (beta, notional, original_entry_date)
    if ("A", "B") in passed:
        s = passed[("A", "B")]
        if (s.beta_original == 0.8 and s.notional == 20000.0
                and s.original_entry_date == pd.Timestamp("2023-01-01")
                and s.fold_origin == 1):
            _passed("frozen fields (beta, notional, original_entry_date, fold_origin) "
                    "preserved")
        else:
            _failed(f"frozen fields mutated: beta={s.beta_original}, "
                    f"notional={s.notional}, oed={s.original_entry_date}, "
                    f"fold_origin={s.fold_origin}")


# ============================================================================
# T5 - extract_open_at_eom (THE BUG-EXPOSING TEST)
# ============================================================================

def _mock_pair_result(
    n_bars: int = 20,
    end_position: int = 1,
    entry_bar: int = 5,
    notional: float = 20_000.0,
    original_entry_date: pd.Timestamp | None = None,
    carries_done: int = 0,
    fold_origin: int = 1,
) -> pd.DataFrame:
    """Construct a fake engine output DataFrame for one pair."""
    dates = pd.bdate_range("2023-01-03", periods=n_bars)
    pos = np.zeros(n_bars, dtype=np.int8)
    if end_position != 0:
        pos[entry_bar:] = end_position
    df = pd.DataFrame({
        "position": pos,
        "daily_pnl_gross": np.zeros(n_bars),
        "daily_pnl_net": np.zeros(n_bars),
        "cum_pnl": np.zeros(n_bars),
        "cost_entry": np.zeros(n_bars),
        "cost_exit": np.zeros(n_bars),
        "borrow_cost": np.zeros(n_bars),
    }, index=dates)
    df.attrs["notional"] = notional
    df.attrs["original_entry_date"] = original_entry_date
    df.attrs["carries_done"] = carries_done
    df.attrs["fold_origin"] = fold_origin
    return df


def t5_extract_open_at_eom():
    print("\nT5 -- extract_open_at_eom (BUG-HUNT for original_entry_date)")

    pairs_df_used = pd.DataFrame({
        "ticker_a": ["X", "Y", "Z"],
        "ticker_b": ["A", "B", "C"],
        "beta_pca": [1.0, 1.5, 2.0],
    })

    # T5a - flat position is NOT extracted (V4 default)
    pair_results = {
        ("X", "A"): _mock_pair_result(end_position=0, entry_bar=5),
    }
    candidates = extract_open_at_eom(pair_results, pairs_df_used, fold_n=1)
    if ("X", "A") not in candidates:
        _passed("flat-at-EOM pair not in candidates")
    else:
        _failed("flat-at-EOM pair leaked into candidates")

    # T5b - open position IS extracted
    pair_results = {
        ("X", "A"): _mock_pair_result(end_position=1, entry_bar=5),
    }
    candidates = extract_open_at_eom(pair_results, pairs_df_used, fold_n=1)
    if ("X", "A") in candidates:
        _passed("open-at-EOM pair extracted")
    else:
        _failed("open-at-EOM pair MISSED")

    # T5c - beta looked up from pairs_df_used
    if candidates[("X", "A")].beta_original == 1.0:
        _passed("beta looked up correctly from pairs_df_used")
    else:
        _failed(f"beta lookup wrong: got {candidates[('X','A')].beta_original}, expected 1.0")

    # T5d -- !!! BUG HUNT !!!
    # Fresh entry: attrs["original_entry_date"] = None (set by engine)
    # extract_open_at_eom should fall back to entry_date (the bar where signal flipped)
    # If .get(key, default) returns None (key exists but value is None), this FAILS.
    expected_entry_date = pd.bdate_range("2023-01-03", periods=20)[5]  # entry_bar=5
    s = candidates[("X", "A")]
    if s.original_entry_date == expected_entry_date:
        _passed(f"fresh-entry original_entry_date correctly resolved to "
                f"{expected_entry_date.date()}")
    elif s.original_entry_date is None:
        _failed(
            f"BUG CONFIRMED: fresh-entry original_entry_date = None, "
            f"expected {expected_entry_date.date()}. attrs.get('original_entry_date', "
            f"entry_date) returned None instead of falling back. Borrow accrual "
            f"on first carry will use day-0 of next fold instead of true entry "
            f"date, under-charging ~20-30 calendar days of borrow per first-carry "
            f"position.",
            is_bug=True,
        )
    else:
        _failed(f"original_entry_date={s.original_entry_date}, expected "
                f"{expected_entry_date}")

    # T5e - already-carried pair: attrs has a non-None date, should be used as-is
    carried_oed = pd.Timestamp("2022-12-15")
    pair_results = {
        ("Y", "B"): _mock_pair_result(
            end_position=1, entry_bar=0,  # held from day-0 (carried in)
            original_entry_date=carried_oed,
            carries_done=2,
        ),
    }
    candidates = extract_open_at_eom(pair_results, pairs_df_used, fold_n=3)
    if (("Y", "B") in candidates
            and candidates[("Y", "B")].original_entry_date == carried_oed):
        _passed(f"carried pair: original_entry_date preserved from attrs "
                f"({carried_oed.date()})")
    else:
        _failed(f"carried pair: original_entry_date lost; got "
                f"{candidates[('Y','B')].original_entry_date if ('Y','B') in candidates else 'MISSING'}")


# ============================================================================
# T6 - refund_eom_exit_cost
# ============================================================================

def t6_refund_eom_exit_cost():
    print("\nT6 -- refund_eom_exit_cost")
    dates = pd.bdate_range("2023-01-03", periods=10)
    df = pd.DataFrame({
        "position": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        "daily_pnl_gross": [0, 0, 0, 10, 20, -5, 15, 8, -3, 12],
        "cost_entry": [0, 0, 30, 0, 0, 0, 0, 0, 0, 0],
        "cost_exit": [0, 0, 0, 0, 0, 0, 0, 0, 0, 30],  # force-closed at last bar
        "borrow_cost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],  # accrued at exit
        "cost_spread": [0]*10,
        "cost_impact": [0]*10,
        "cost_commission": [0]*10,
        "daily_pnl_net": [0, 0, -30, 10, 20, -5, 15, 8, -3, 12 - 30 - 5],
    }, index=dates)
    df["cum_pnl"] = df["daily_pnl_net"].cumsum()

    pair_results = {("X", "A"): df}
    passed = {
        ("X", "A"): CarryState(
            ticker_a="X", ticker_b="A", position=1, beta_original=1.0,
            notional=20000.0, original_entry_date=dates[2],
            carries_done=0, fold_origin=1,
        ),
    }

    n_refunded = refund_eom_exit_cost(pair_results, passed)
    if n_refunded == 1:
        _passed("refunded 1 pair")
    else:
        _failed(f"refunded {n_refunded}, expected 1")

    last = pair_results[("X", "A")].iloc[-1]
    if last["cost_exit"] == 0.0 and last["borrow_cost"] == 0.0:
        _passed("last-bar cost_exit and borrow_cost zeroed")
    else:
        _failed(f"refund did not zero costs: cost_exit={last['cost_exit']}, "
                f"borrow={last['borrow_cost']}")

    # T6c - daily_pnl_net at last bar = gross - cost_entry (which is 0 here)
    expected_net_last = 12.0  # gross 12, cost_entry 0, exit + borrow refunded
    if abs(last["daily_pnl_net"] - expected_net_last) < 1e-9:
        _passed(f"daily_pnl_net at last bar recomputed: {last['daily_pnl_net']:.2f}")
    else:
        _failed(f"daily_pnl_net last bar = {last['daily_pnl_net']}, "
                f"expected {expected_net_last}")

    # T6d - cum_pnl is consistent with daily_pnl_net
    cum_expected = pair_results[("X", "A")]["daily_pnl_net"].cumsum()
    cum_actual = pair_results[("X", "A")]["cum_pnl"]
    if (cum_expected.values == cum_actual.values).all():
        _passed("cum_pnl matches cumsum(daily_pnl_net)")
    else:
        _failed("cum_pnl out of sync with daily_pnl_net")


# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Unit/smoke tests for engine_daily/carry_forward.py")
    print("=" * 70)
    t1_carry_state()
    t2_compute_residuals_for_gate()
    t3_gate_test_single()
    t4_apply_gate()
    t5_extract_open_at_eom()
    t6_refund_eom_exit_cost()

    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if BUG_FLAGS:
        print(f"  BUGS CONFIRMED: {len(BUG_FLAGS)}")
        for b in BUG_FLAGS:
            print(f"    - {b[:80]}{'...' if len(b) > 80 else ''}")
    print("=" * 70)
    sys.exit(0 if FAIL_COUNT == 0 else 1)
