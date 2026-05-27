"""
test_pipeline_smoke.py — End-to-end 3-fold carry-forward smoke
==============================================================

Drives the full V4 + carry-forward pipeline on a tiny synthetic universe
(~30 tickers, ~15 months of daily bars). Uses the real `run_fold_daily`,
real `discovery_daily.run`, real `apply_gate` / `extract_open_at_eom` /
`refund_eom_exit_cost` -- everything except the daily cache (which we
synthesize).

This catches integration bugs that unit tests miss:
    I1 - carry_state is non-empty across at least one fold boundary
    I2 - n_trades = zero_cross + hard_sl + open_at_eom invariant per fold
    I3 - carried positions have initial_position != 0 in next fold's engine
    I4 - frozen beta is actually used by next fold's engine (not overridden by
         discovery's beta when the pair is re-discovered) -- COMPLEMENTS the
         standalone audit_beta_source_split.py
    I5 - cum_pnl is monotonic in cumsum of daily_pnl_net (refund correctness)
    I6 - no double-booked entry costs (entry cost on carried pair's day-0
         should be 0)

Run: python audits/carryforward_audit/test_pipeline_smoke.py
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
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))

from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily
from engine_daily.metrics_daily import aggregate_fold_metrics
from engine_daily.carry_forward import (
    apply_gate, compute_residuals_for_gate,
    extract_open_at_eom, refund_eom_exit_cost,
    GATE_PVAL_THRESHOLD,
)
from engine.phase1_cointegration.factor_residual import project_residual

PASS_COUNT = 0
FAIL_COUNT = 0
FINDINGS: list[str] = []


def _passed(msg: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}", flush=True)


def _failed(msg: str, *, finding: bool = False) -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {msg}", flush=True)
    if finding:
        FINDINGS.append(msg)


def _flagged(msg: str) -> None:
    """Informational only -- not a pass or fail."""
    print(f"  [FLAG] {msg}", flush=True)


# ============================================================================
# Synthetic universe builder: ~30 tickers, ~15 months, with embedded
# cointegrated pairs that persist across folds.
# ============================================================================

def _synth_universe(
    n_tickers: int = 60,
    n_days: int = 320,  # ~15 months
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    market = np.cumsum(rng.normal(0, 0.012, n_days))

    # Build 3 sector factors (slow processes) — these create within-sector
    # cointegratable structure that survives 1-month re-discoveries.
    sector_factors = [np.cumsum(rng.normal(0, 0.004, n_days)) for _ in range(3)]

    out: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        sector_id = i % 3
        beta_mkt = 0.5 + 0.6 * rng.random()
        beta_sec = 0.4 + 0.4 * rng.random()
        idio = np.cumsum(rng.normal(0, 0.006, n_days))
        log_close = (4.0
                     + beta_mkt * market
                     + beta_sec * sector_factors[sector_id]
                     + idio)
        vol = rng.integers(1_000_000, 8_000_000, n_days).astype(float)
        out[f"T{i:03d}"] = pd.DataFrame(
            {"log_close": log_close, "volume": vol}, index=dates,
        )
    return out


def _slice(cache, start_ts, end_ts):
    out = {}
    for tk, df in cache.items():
        s = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(s) > 0:
            out[tk] = s
    return out


# ============================================================================
# 3-fold walk-forward driver (mirrors scripts/run_v4_pipeline.py logic)
# ============================================================================

def _run_3_folds(cache: dict, use_carry: bool) -> list[dict]:
    """Run a 3-fold synthetic walk-forward with `--use-carry-forward` mode toggle.

    Returns a list of per-fold result dicts incl. fold_metrics, pair_results,
    and carry-forward diagnostics.
    """
    # Form 12 months / trade 1 month, walked forward by 1 month per fold.
    fold_schedule = [
        (1, "2022-01-03", "2022-12-30", "2023-01-03", "2023-01-31"),
        (2, "2022-02-01", "2023-01-31", "2023-02-01", "2023-02-28"),
        (3, "2022-03-01", "2023-02-28", "2023-03-01", "2023-03-31"),
    ]
    # Need next-fold formation for gate. Provide a "fold 4" formation that the
    # gate at end-of-fold-3 will use. (Fold 4 itself doesn't trade in this smoke.)
    next_form_for = {
        1: ("2022-02-01", "2023-01-31"),
        2: ("2022-03-01", "2023-02-28"),
        3: ("2022-04-01", "2023-03-31"),
    }

    results = []
    carry_state: dict = {}

    for fold_n, fs, fe, ts, te in fold_schedule:
        n_carry_in = len(carry_state)

        formation = _slice(cache, pd.Timestamp(fs), pd.Timestamp(fe))
        trading = _slice(cache, pd.Timestamp(ts), pd.Timestamp(te))

        pairs_df, factor_state = discovery_daily.run(
            formation_data=formation, hl_min=5.0, hl_max=30.0,
        )

        if pairs_df.empty or not factor_state:
            results.append({"fold": fold_n, "skip": "empty discovery",
                            "n_carry_in": n_carry_in, "n_carry_out": 0,
                            "n_gate_fail": n_carry_in})
            if use_carry:
                carry_state = {}
            continue

        loadings_W = factor_state["loadings_W"]
        factor_tickers = factor_state["tickers"]
        resid_form_df = factor_state["residual_log_prices"]
        resid_trade_dict = project_residual(
            trading, loadings_W, factor_tickers, min_obs=10,
        )
        # Path A re-anchor (same as production pipeline)
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form_df.columns:
                continue
            s_form_tk = resid_form_df[tk].dropna()
            if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
                continue
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
        resid_trade_df = pd.concat(resid_trade_dict, axis=1)

        # Engine call — passes carry_state_in only if use_carry
        pair_results = run_fold_daily(
            pairs_df=pairs_df,
            resid_form=resid_form_df,
            resid_trade=resid_trade_df,
            alpha_lookback=60,
            entry_z=2.0, z_window=60, hard_sl_z=4.0,
            cost_data=None,  # flat-cost (no Week-5 spread cache needed)
            carry_state_in=(carry_state if use_carry else None),
            current_fold_n=fold_n,
        )

        fold_metrics = aggregate_fold_metrics(pair_results, total_capital=1_000_000.0)

        # ---- E2 gate test for next fold ----
        n_carry_out = 0
        n_gate_fail = 0
        if use_carry:
            # Mirror pipeline's beta_lookup_df build
            beta_records = [
                {"ticker_a": r["ticker_a"], "ticker_b": r["ticker_b"],
                 "beta_pca": float(r["beta_pca"])}
                for _, r in pairs_df.iterrows()
            ]
            existing_keys = {(r["ticker_a"], r["ticker_b"]) for r in beta_records}
            for key, state in carry_state.items():
                if key not in existing_keys:
                    beta_records.append({
                        "ticker_a": state.ticker_a, "ticker_b": state.ticker_b,
                        "beta_pca": float(state.beta_original),
                    })
            beta_lookup_df = pd.DataFrame(beta_records)
            candidates = extract_open_at_eom(pair_results, beta_lookup_df, fold_n)

            fs_next, fe_next = next_form_for[fold_n]
            next_formation = _slice(cache, pd.Timestamp(fs_next), pd.Timestamp(fe_next))
            next_resid_df, _ = compute_residuals_for_gate(next_formation)

            passed, pvals = apply_gate(candidates, next_resid_df,
                                       gate_threshold=GATE_PVAL_THRESHOLD)
            n_carry_out = len(passed)
            n_gate_fail = len(candidates) - n_carry_out

            if passed:
                refund_eom_exit_cost(pair_results, passed)

            carry_state = passed

        # Recompute metrics after refund
        fold_metrics = aggregate_fold_metrics(pair_results, total_capital=1_000_000.0)

        results.append({
            "fold": fold_n,
            "pairs_df": pairs_df,
            "pair_results": pair_results,
            "fold_metrics": fold_metrics,
            "n_pairs": len(pairs_df),
            "n_carry_in": n_carry_in,
            "n_carry_out": n_carry_out,
            "n_gate_fail": n_gate_fail,
            "carry_state_out": dict(carry_state) if use_carry else {},
        })

    return results


# ============================================================================
# Invariant checks
# ============================================================================

def run_smoke():
    print("=" * 72)
    print("  test_pipeline_smoke.py — 3-fold E2E with carry-forward")
    print("=" * 72)

    cache = _synth_universe(n_tickers=60, n_days=320, seed=7)
    print(f"\n  Synth universe: {len(cache)} tickers, "
          f"{len(next(iter(cache.values())))} daily bars")

    # Run with carry-forward ON
    print("\n--- 3-fold run with --use-carry-forward ---")
    try:
        results_cf = _run_3_folds(cache, use_carry=True)
    except Exception as e:
        _failed(f"carry-forward run crashed: {e}", finding=True)
        traceback.print_exc()
        return

    # Run with carry-forward OFF (V4 baseline)
    print("\n--- 3-fold run with V4 baseline (no carry-forward) ---")
    try:
        results_baseline = _run_3_folds(cache, use_carry=False)
    except Exception as e:
        _failed(f"baseline run crashed: {e}", finding=True)
        traceback.print_exc()
        return

    # ============================================================
    # I1 - carry_state non-empty across some boundary
    # ============================================================
    print("\nI1 -- carry-state propagates across fold boundary")
    any_carry = any(r.get("n_carry_out", 0) > 0 for r in results_cf)
    if any_carry:
        for r in results_cf:
            if r.get("n_carry_out", 0) > 0:
                _passed(f"fold {r['fold']}: carry_out={r['n_carry_out']} "
                        f"(of {r.get('n_carry_in',0)+r.get('n_carry_out',0)+r.get('n_gate_fail',0)} candidates)")
                break
    else:
        _flagged("no carry-forward state propagated in 3-fold smoke. Synthetic "
                 "universe may not have produced any open-at-EOM positions. "
                 "Not a bug — just means the rest of the carry-specific checks "
                 "are uninformative.")

    # ============================================================
    # I2 - per-fold trade count invariant
    # ============================================================
    print("\nI2 -- n_trades = zero_cross + hard_sl + open_at_eom invariant")
    for r in results_cf:
        if "skip" in r:
            continue
        fm = r["fold_metrics"]
        eb = fm["exit_breakdown"]
        expected = eb["zero_cross"] + eb["hard_sl"] + eb["open_at_eom"]
        if fm["n_trades"] == expected:
            _passed(f"fold {r['fold']}: n_trades={fm['n_trades']} = "
                    f"zc({eb['zero_cross']}) + sl({eb['hard_sl']}) + "
                    f"eom({eb['open_at_eom']})")
        else:
            _failed(f"fold {r['fold']}: invariant broken: "
                    f"n_trades={fm['n_trades']} != {expected} "
                    f"(zc+sl+eom = {eb['zero_cross']}+{eb['hard_sl']}+{eb['open_at_eom']})",
                    finding=True)

    # ============================================================
    # I3 - carry-forward sets initial_position!=0 for carried pairs
    # ============================================================
    print("\nI3 -- carried pairs have initial_position!=0 in next fold's engine")
    found = False
    for prev_r, curr_r in zip(results_cf[:-1], results_cf[1:]):
        if not prev_r.get("carry_state_out"):
            continue
        if "pair_results" not in curr_r:
            continue
        for key in prev_r["carry_state_out"]:
            if key in curr_r["pair_results"]:
                pdf = curr_r["pair_results"][key]
                init = pdf.attrs.get("initial_position", 0)
                if init != 0:
                    _passed(f"fold {curr_r['fold']}, pair {key}: "
                            f"initial_position={init} (carried from fold "
                            f"{prev_r['fold']})")
                    found = True
                    break
                else:
                    _failed(f"fold {curr_r['fold']}, pair {key}: "
                            f"initial_position=0 but pair was carried in!",
                            finding=True)
        if found:
            break
    if not found and not any_carry:
        _flagged("skipped (no carries to verify)")

    # ============================================================
    # I4 - frozen beta actually used in next fold (NOT discovery's beta)
    # ============================================================
    print("\nI4 -- frozen beta preserved across boundary")
    for prev_r, curr_r in zip(results_cf[:-1], results_cf[1:]):
        if not prev_r.get("carry_state_out"):
            continue
        for key, state in prev_r["carry_state_out"].items():
            if key in curr_r.get("pair_results", {}):
                # Inspect curr fold's CarryState in carry_state_out for this pair
                next_state = curr_r["carry_state_out"].get(key)
                if next_state is not None:
                    # If the pair is in curr's pairs_df with a DIFFERENT beta,
                    # next_state.beta_original SHOULD == state.beta_original (frozen)
                    pairs_df_curr = curr_r["pairs_df"]
                    matching = pairs_df_curr[
                        (pairs_df_curr["ticker_a"] == key[0])
                        & (pairs_df_curr["ticker_b"] == key[1])
                    ]
                    if not matching.empty:
                        beta_disc = float(matching["beta_pca"].iloc[0])
                        if abs(state.beta_original - beta_disc) > 1e-6:
                            # The pair was re-discovered with different beta.
                            # next_state.beta_original SHOULD be state.beta_original.
                            if abs(next_state.beta_original - state.beta_original) < 1e-9:
                                _passed(
                                    f"pair {key}: beta preserved across boundary "
                                    f"(beta_carry={state.beta_original:.4f}, "
                                    f"beta_discovered={beta_disc:.4f})"
                                )
                            elif abs(next_state.beta_original - beta_disc) < 1e-9:
                                _failed(
                                    f"BUG: pair {key} re-discovered with beta="
                                    f"{beta_disc:.4f}, but next CarryState beta="
                                    f"{next_state.beta_original:.4f} (should be "
                                    f"frozen at {state.beta_original:.4f})",
                                    finding=True,
                                )
                            else:
                                _failed(
                                    f"pair {key}: beta unexpected value "
                                    f"{next_state.beta_original:.4f}",
                                    finding=True,
                                )
                            break
        else:
            continue
        break

    # ============================================================
    # I5 - cum_pnl = cumsum(daily_pnl_net) for every pair
    # ============================================================
    print("\nI5 -- cum_pnl is consistent with cumsum(daily_pnl_net)")
    n_checked = 0
    n_violated = 0
    for r in results_cf:
        if "pair_results" not in r:
            continue
        for key, pdf in r["pair_results"].items():
            if pdf.empty:
                continue
            n_checked += 1
            cum_expected = pdf["daily_pnl_net"].cumsum().values
            cum_actual = pdf["cum_pnl"].values
            if not np.allclose(cum_expected, cum_actual, atol=1e-9):
                n_violated += 1
    if n_violated == 0:
        _passed(f"checked {n_checked} pairs; all cum_pnl consistent with cumsum")
    else:
        _failed(f"{n_violated} / {n_checked} pairs have cum_pnl mismatch "
                f"(refund recomputation broken)", finding=True)

    # ============================================================
    # I6 - no entry cost booked on day-0 of carried positions
    # ============================================================
    print("\nI6 -- no entry cost booked on day-0 of carried positions")
    n_carry_pairs_checked = 0
    n_with_day0_entry_cost = 0
    for prev_r, curr_r in zip(results_cf[:-1], results_cf[1:]):
        if not prev_r.get("carry_state_out"):
            continue
        for key in prev_r["carry_state_out"]:
            if key in curr_r.get("pair_results", {}):
                pdf = curr_r["pair_results"][key]
                if pdf.empty:
                    continue
                n_carry_pairs_checked += 1
                day0_entry = float(pdf["cost_entry"].iloc[0])
                if day0_entry != 0.0:
                    n_with_day0_entry_cost += 1
                    _failed(
                        f"pair {key} carried into fold {curr_r['fold']} has "
                        f"cost_entry[0]={day0_entry} (should be 0; entry cost "
                        f"was paid in fold {prev_r['fold']})",
                        finding=True,
                    )
    if n_carry_pairs_checked == 0:
        _flagged("skipped (no carries to verify)")
    elif n_with_day0_entry_cost == 0:
        _passed(f"checked {n_carry_pairs_checked} carried pairs; none have "
                f"day-0 entry cost. (Entry already paid in origin fold.)")

    # ============================================================
    # Summary diagnostic - carry-forward vs baseline aggregate metrics
    # ============================================================
    print("\n--- Diagnostic: carry-forward vs baseline aggregate (smoke) ---")
    for r_cf, r_bl in zip(results_cf, results_baseline):
        if "skip" in r_cf or "skip" in r_bl:
            continue
        fm_cf = r_cf["fold_metrics"]
        fm_bl = r_bl["fold_metrics"]
        eb_cf = fm_cf["exit_breakdown"]
        eb_bl = fm_bl["exit_breakdown"]
        print(f"  fold {r_cf['fold']:>2}: "
              f"V4=({fm_bl['n_trades']:>3}t, sh={fm_bl['sharpe']:+.2f}, "
              f"eom={eb_bl['open_at_eom']:>3}) | "
              f"CF=({fm_cf['n_trades']:>3}t, sh={fm_cf['sharpe']:+.2f}, "
              f"eom={eb_cf['open_at_eom']:>3}, carry_out={r_cf.get('n_carry_out',0)})")


# ============================================================
if __name__ == "__main__":
    run_smoke()
    print("\n" + "=" * 72)
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FINDINGS:
        print(f"  BUGS / INVARIANT VIOLATIONS: {len(FINDINGS)}")
        for f in FINDINGS:
            print(f"    - {f[:90]}{'...' if len(f) > 90 else ''}")
    print("=" * 72)
    sys.exit(0 if FAIL_COUNT == 0 else 1)
