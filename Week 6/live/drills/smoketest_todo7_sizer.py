"""TODO 7 smoketest: vol-target sizer.

Cross-path: live wrapper output = backtest compute_vol_target_notional_daily
for same spread, same params.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine_daily.engine_daily import compute_vol_target_notional_daily
from live.engine_live.alpha_refit import load_and_refit
from live.engine_live.sizer import (
    _PAPER_CAP, _PAPER_FLOOR, _PAPER_TARGET_DAILY,
    compute_pair_notional, size_all_pairs,
)

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_compute_pair_notional_basic():
    """Synthetic spread → notional within [floor, cap]."""
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 0.5, 252).cumsum()
    n = compute_pair_notional(spread)
    check("notional finite", np.isfinite(n))
    check(f"notional in paper bounds [{_PAPER_FLOOR}, {_PAPER_CAP}]",
          _PAPER_FLOOR <= n <= _PAPER_CAP, f"got {n:.2f}")


def t_floor_clamp_on_low_vol():
    """Constant spread (sigma=0) → notional = CAP (per backtest semantics)."""
    spread = np.ones(252) * 5.0
    n = compute_pair_notional(spread)
    check("constant spread (sigma~0): notional clamps to CAP",
          abs(n - _PAPER_CAP) < 1e-9, f"got {n}")


def t_cap_clamp_on_high_vol():
    """High-vol spread → notional clamps to FLOOR."""
    rng = np.random.default_rng(1)
    spread = rng.normal(0, 100.0, 252).cumsum()   # very high vol
    n = compute_pair_notional(spread)
    check("high-vol spread: notional clamps to FLOOR",
          abs(n - _PAPER_FLOOR) < 1e-9, f"got {n}")


def t_cross_path_vs_backtest():
    """Live compute_pair_notional == backtest compute_vol_target_notional_daily
    for identical inputs."""
    rng = np.random.default_rng(42)
    for seed in range(5):
        spread = rng.normal(0, rng.uniform(0.1, 2.0), 252).cumsum()
        live = compute_pair_notional(spread,
                                     target_daily_vol=_PAPER_TARGET_DAILY,
                                     floor=_PAPER_FLOOR, cap=_PAPER_CAP)
        bt = compute_vol_target_notional_daily(spread,
                                               target_daily_vol=_PAPER_TARGET_DAILY,
                                               floor=_PAPER_FLOOR, cap=_PAPER_CAP)
        check(f"seed {seed}: live == backtest", abs(live - bt) < 1e-12,
              f"live={live} vs bt={bt}")


def t_size_all_pairs_real_data():
    """Run on real discovery output. All sizes finite + within bounds."""
    refits = load_and_refit(PAIRS_FP, FACTOR_FP, n_lookback=60)
    alphas = {(r.ticker_a, r.ticker_b): r.alpha_refit for r in refits}
    pairs_df = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    sizes = size_all_pairs(pairs_df, fs["residual_log_prices"], alphas)
    check(f"size_all_pairs returns {len(pairs_df)} entries",
          len(sizes) == len(pairs_df),
          f"got {len(sizes)} vs {len(pairs_df)} pairs")
    out_of_bounds = [s for s in sizes
                     if not (_PAPER_FLOOR <= s.notional <= _PAPER_CAP)]
    check(f"all notionals within [{_PAPER_FLOOR}, {_PAPER_CAP}]",
          len(out_of_bounds) == 0, f"violations: {len(out_of_bounds)}")


def t_total_gross_within_margin():
    """13 simultaneous pairs * 2 legs * cap = gross exposure check.

    Paper account $100k * Reg T 2x = $200k buying power. Need 13 pairs * 2 legs * cap
    <= $200k. With cap=$4k: 13 * 2 * 4000 = $104k <= $200k. Safe.
    """
    max_simultaneous = 13   # per documented paper margin analysis
    legs_per_pair = 2
    worst_case_gross = max_simultaneous * legs_per_pair * _PAPER_CAP
    margin_buying_power = 200_000.0    # $100k cash * Reg T 2x
    check(f"worst-case gross ({max_simultaneous} pairs * 2 legs * cap) "
          f"<= buying power ${margin_buying_power:.0f}",
          worst_case_gross <= margin_buying_power,
          f"worst_case=${worst_case_gross:.0f}, headroom=${margin_buying_power - worst_case_gross:.0f}")


def t_paper_scale_vs_backtest_scale():
    """Confirm paper defaults are 10x scaled from backtest defaults."""
    backtest_target = 200.0
    backtest_floor = 10_000.0
    backtest_cap = 40_000.0
    check("paper target_daily_vol = backtest / 10",
          abs(_PAPER_TARGET_DAILY - backtest_target / 10) < 1e-9,
          f"paper={_PAPER_TARGET_DAILY}, backtest/10={backtest_target/10}")
    check("paper floor = backtest / 10",
          abs(_PAPER_FLOOR - backtest_floor / 10) < 1e-9)
    check("paper cap = backtest / 10",
          abs(_PAPER_CAP - backtest_cap / 10) < 1e-9)


def t_hardstop_still_works():
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t7_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo7")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 7 Smoketest: vol-target sizer ==\n")
    print("--- Basic sizing ---")
    t_compute_pair_notional_basic()
    print("\n--- Floor clamp (constant spread) ---")
    t_floor_clamp_on_low_vol()
    print("\n--- Cap clamp (high vol) ---")
    t_cap_clamp_on_high_vol()
    print("\n--- Cross-path vs backtest ---")
    t_cross_path_vs_backtest()
    print("\n--- size_all_pairs on real pair list ---")
    t_size_all_pairs_real_data()
    print("\n--- Total gross within paper margin ---")
    t_total_gross_within_margin()
    print("\n--- Paper scale 10x backtest ---")
    t_paper_scale_vs_backtest_scale()
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
