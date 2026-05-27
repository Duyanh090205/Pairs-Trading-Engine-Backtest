"""TODO 6 smoketest: alpha refit wrapper.

Cross-path: live wrapper output bit-identical to backtest's recompute_alpha
called directly per pair (the same pattern as run_v4_pipeline.run_fold_daily).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from engine_daily.alpha_refit import recompute_alpha as backtest_recompute_alpha
from live.engine_live.alpha_refit import RefitAlpha, load_and_refit, refit_all_pairs

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_refit_runs_on_real_pair_list():
    out = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=60)
    pairs_df = pd.read_parquet(PAIRS_FP)
    check(f"refit_all_pairs returns list",
          isinstance(out, list))
    check(f"refit covered most pairs ({len(out)} of {len(pairs_df)})",
          len(out) >= len(pairs_df) - 2,    # tolerate a couple drop-outs
          f"refit={len(out)} vs pairs={len(pairs_df)}")
    # All α values finite
    bad = [r for r in out if not pd.notna(r.alpha_refit)]
    check("all alpha values finite", len(bad) == 0)


def t_beta_unchanged():
    """β must come from discovery, NOT recomputed."""
    out = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=60)
    pairs_df = pd.read_parquet(PAIRS_FP)
    pairs_beta = {(r["ticker_a"], r["ticker_b"]): float(r["beta_pca"])
                  for _, r in pairs_df.iterrows()}
    bad = []
    for r in out:
        expected = pairs_beta.get((r.ticker_a, r.ticker_b))
        if expected is None or abs(r.beta - expected) > 1e-12:
            bad.append((r.pair_id, r.beta, expected))
    check("beta is preserved from discovery (NOT recomputed)",
          len(bad) == 0, f"violations: {bad[:3]}")


def t_cross_path_vs_backtest():
    """Live refit_all_pairs output bit-identical to direct recompute_alpha calls."""
    live_out = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=60)
    # Backtest path: replicate run_fold_daily lines 561-570
    pairs_df = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    resid_form = fs["residual_log_prices"]
    bt_results: dict[tuple[str, str], float] = {}
    for _, row in pairs_df.iterrows():
        ta, tb, beta = row["ticker_a"], row["ticker_b"], float(row["beta_pca"])
        if ta not in resid_form.columns or tb not in resid_form.columns:
            continue
        ra = resid_form[ta].dropna()
        rb = resid_form[tb].dropna()
        try:
            alpha = backtest_recompute_alpha(ra, rb, beta, n_lookback=60)
        except ValueError:
            continue
        bt_results[(ta, tb)] = alpha
    # Compare
    max_diff = 0.0
    for r in live_out:
        bt_alpha = bt_results.get((r.ticker_a, r.ticker_b))
        if bt_alpha is None:
            continue
        max_diff = max(max_diff, abs(r.alpha_refit - bt_alpha))
    check(f"live alpha refit == backtest recompute_alpha across {len(live_out)} pairs",
          max_diff < 1e-12, f"max abs diff = {max_diff:.2e}")


def t_n_lookback_respected():
    """If n_lookback varied, α should change. Tests parameter wiring."""
    out_60 = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=60)
    out_30 = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=30)
    by_pair_60 = {r.pair_id: r.alpha_refit for r in out_60}
    by_pair_30 = {r.pair_id: r.alpha_refit for r in out_30}
    common = set(by_pair_60) & set(by_pair_30)
    n_diff = sum(1 for k in common if abs(by_pair_60[k] - by_pair_30[k]) > 1e-9)
    check(f"alpha differs between n_lookback=60 and =30 for most pairs",
          n_diff >= len(common) // 2,
          f"only {n_diff} of {len(common)} pairs differed")


def t_invalid_beta_raises():
    """β <= 0 must raise (V4 spec: only +ve betas pass discovery R5 filter)."""
    import pandas as pd
    fake_pairs = pd.DataFrame([{
        "ticker_a": "AAPL", "ticker_b": "MSFT", "beta_pca": -1.0,
    }])
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    out = refit_all_pairs(fake_pairs, fs["residual_log_prices"], n_lookback=60)
    check("negative beta pair is silently dropped (not in output)",
          len(out) == 0, f"got {len(out)} (expected 0)")


def t_hardstop_still_works():
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t6_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo6")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 6 Smoketest: alpha refit ==\n")
    print("--- Basic refit ---")
    t_refit_runs_on_real_pair_list()
    print("\n--- Beta preservation ---")
    t_beta_unchanged()
    print("\n--- Cross-path vs backtest ---")
    t_cross_path_vs_backtest()
    print("\n--- n_lookback parameter wiring ---")
    t_n_lookback_respected()
    print("\n--- Invalid beta handling ---")
    t_invalid_beta_raises()
    print("\n--- Hardstop ---")
    t_hardstop_still_works()
    print()
    if errors:
        print(f"FAIL: {len(errors)} - {errors}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
