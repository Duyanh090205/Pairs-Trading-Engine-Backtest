"""
audit_beta_source_split.py — Bug-hunt for beta-source mismatch
============================================================

PATTERN: integration-layer wiring bug (per /deep-audit-bug skill).

SUSPECTED BUG:
    When a pair (a, b) is BOTH:
       - in fold N+1's discovery output (pairs_df), with discovery beta_disc
       - in carry_state_in (carried from fold N), with frozen beta_carry

    The engine `run_fold_daily` (engine_daily.py:545-550) executes the pair
    with beta_carry (frozen) -- correct per pre-commit.

    But the pipeline's `run_v4_pipeline.py:384-397` builds beta_lookup_df from
    `pairs_df_used` (= discovery's pairs_df, NOT the effective list), and the
    `if key not in existing_keys` guard prevents overwriting with the frozen beta.

    Then `extract_open_at_eom` uses beta_lookup_df to build the NEXT fold's
    CarryState. So the new CarryState.beta_original is beta_disc, not beta_carry.

    Net effect: next fold's engine sees beta = beta_disc, opens / continues the
    "same" position with a DIFFERENT hedge ratio than it was actually
    holding. The frozen-beta invariant is violated at every fold boundary where
    a carry pair was re-discovered.

EXPECTED FAILURE MODE:
    Test below builds a synthetic scenario with beta_carry=2.0 and beta_disc=3.0.
    extract_open_at_eom returns a CarryState with beta=3.0 (discovery's),
    confirming the bug. If it returns 2.0, the bug is not present.
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

from engine_daily.carry_forward import CarryState, extract_open_at_eom


def main() -> int:
    print("=" * 72)
    print("  audit_beta_source_split.py")
    print("  Suspected bug: beta_carry vs beta_discovery split for re-discovered carry pairs")
    print("=" * 72)

    BETA_CARRY = 2.0       # frozen beta from prior fold when position was opened
    BETA_DISCOVERY = 3.0   # beta computed by fold N+1's own discovery (different!)
    PAIR = ("XYZ", "ABC")

    # ---- Simulate what the pipeline builds ----
    # pairs_df_used is the discovery output for fold N+1 — has the SAME pair (a, b)
    # back in the list but with a freshly-estimated beta = beta_discovery.
    pairs_df_used = pd.DataFrame({
        "ticker_a": [PAIR[0], "OTHER"],
        "ticker_b": [PAIR[1], "OTHER2"],
        "beta_pca": [BETA_DISCOVERY, 1.0],
    })

    # The position is "open at EOM" in fold N+1; this is the engine output.
    # The engine USED beta_carry (carry override in engine_daily.py:548) — that's
    # the hedge ratio actually traded. We mock this by NOT explicitly storing beta
    # in pair_results (which is realistic — engine output doesn't carry beta as a
    # column; it's only in attrs / pairs_df_used).
    dates = pd.bdate_range("2023-02-01", periods=20)
    pos = np.zeros(20, dtype=np.int8)
    pos[:] = 1   # carried from prior fold; held all of fold N+1 (open at EOM)
    pdf = pd.DataFrame({
        "position": pos,
        "daily_pnl_gross": np.zeros(20),
        "daily_pnl_net": np.zeros(20),
        "cum_pnl": np.zeros(20),
        "cost_entry": np.zeros(20),
        "cost_exit": np.zeros(20),
        "borrow_cost": np.zeros(20),
    }, index=dates)
    # Carry attrs (set by engine_daily.run_pair_daily when initial_position != 0)
    pdf.attrs["notional"] = 20000.0
    pdf.attrs["original_entry_date"] = pd.Timestamp("2023-01-15")
    pdf.attrs["carries_done"] = 1
    pdf.attrs["fold_origin"] = 1
    pair_results = {PAIR: pdf}

    # ---- Now simulate the pipeline's beta_lookup_df build ----
    # This mirrors run_v4_pipeline.py:384-397 verbatim.
    beta_records = [
        {"ticker_a": r["ticker_a"], "ticker_b": r["ticker_b"],
         "beta_pca": float(r["beta_pca"])}
        for _, r in pairs_df_used.iterrows()
    ]

    # CarryState the engine ACTUALLY used (beta_carry = 2.0)
    carry_state_in = {
        PAIR: CarryState(
            ticker_a=PAIR[0], ticker_b=PAIR[1], position=1,
            beta_original=BETA_CARRY, notional=20000.0,
            original_entry_date=pd.Timestamp("2023-01-15"),
            carries_done=0, fold_origin=1,
        ),
    }

    # The pipeline's "inject only if not already in beta_records" logic
    existing_keys = {(r["ticker_a"], r["ticker_b"]) for r in beta_records}
    for key, state in carry_state_in.items():
        if key not in existing_keys:
            beta_records.append({
                "ticker_a": state.ticker_a, "ticker_b": state.ticker_b,
                "beta_pca": float(state.beta_original),
            })
    beta_lookup_df = pd.DataFrame(beta_records)

    # ---- extract_open_at_eom builds the NEXT fold's CarryState ----
    new_carry = extract_open_at_eom(pair_results, beta_lookup_df, fold_n=2)

    print("\n  Setup:")
    print(f"    Pair                                : {PAIR}")
    print(f"    beta actually used by engine (frozen)  : {BETA_CARRY}")
    print(f"    beta in pairs_df_used (discovery's)    : {BETA_DISCOVERY}")
    print()

    if PAIR not in new_carry:
        print(f"  [FAIL] Pair not in new_carry; cannot evaluate the bug.")
        return 1

    extracted_beta = new_carry[PAIR].beta_original
    print(f"  extract_open_at_eom returned beta        : {extracted_beta}")
    print()

    if extracted_beta == BETA_CARRY:
        print(f"  [PASS] beta preserved (frozen-beta invariant holds across fold boundary).")
        return 0
    elif extracted_beta == BETA_DISCOVERY:
        print(f"  [BUG CONFIRMED — HIGH CONFIDENCE]")
        print(f"  ---------------------------------")
        print(f"  Pipeline propagates beta_discovery into the next fold's CarryState,")
        print(f"  but the engine actually executed with beta_carry. The frozen-beta")
        print(f"  invariant is broken at every fold boundary where a carry pair")
        print(f"  is also re-discovered.")
        print()
        print(f"  Impact:")
        print(f"    - Position held with hedge ratio drift (beta_carry -> beta_discovery")
        print(f"      at each re-discovery boundary).")
        print(f"    - Violates the pre-committed 'beta frozen at original entry'")
        print(f"      policy locked in log.md on 2026-05-24.")
        print(f"    - Effect proportional to how often carry pairs ALSO clear")
        print(f"      fold N+1's BH-FDR. Hard to bound without running the audit")
        print(f"      against real pipeline output.")
        print()
        print(f"  Location:")
        print(f"    Producer A: pipeline beta_records (run_v4_pipeline.py:~384)")
        print(f"                -> uses pairs_df_used (discovery's beta)")
        print(f"    Producer B: carry_state.beta_original (engine_daily.py:~548)")
        print(f"                -> uses frozen beta")
        print(f"    Consumer:   extract_open_at_eom (carry_forward.py:~205)")
        print(f"                -> picks producer A when both available")
        print()
        print(f"  Fix (surgical):")
        print(f"    In run_v4_pipeline.py at the beta_lookup_df build, OVERWRITE")
        print(f"    discovery's beta with carry's beta whenever a pair is in BOTH:")
        print()
        print(f"      # Replace the current 'if key not in existing_keys: append' block")
        print(f"      beta_records_dict = {{(r['ticker_a'], r['ticker_b']): float(r['beta_pca'])")
        print(f"                          for _, r in pairs_df_used.iterrows()}}")
        print(f"      for key, state in carry_state.items():")
        print(f"          beta_records_dict[key] = float(state.beta_original)  # always override")
        print(f"      beta_lookup_df = pd.DataFrame([")
        print(f"          {{'ticker_a': k[0], 'ticker_b': k[1], 'beta_pca': v}}")
        print(f"          for k, v in beta_records_dict.items()")
        print(f"      ])")
        return 1
    else:
        print(f"  [FAIL] beta unexpected value: {extracted_beta} "
              f"(expected {BETA_CARRY} or {BETA_DISCOVERY})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
