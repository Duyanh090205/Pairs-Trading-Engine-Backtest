"""
Audit: cost engine integration (Week 5 dynamic) vs flat-cost baseline.

Three things to verify:

  A. UNIT SANITY — cost_engine primitives produce expected magnitudes on
     hand-checked synthetic inputs (no surprises in formula units).

  B. LOOKUP HIT RATE — when the V4 engine asks for spread on (ticker, date),
     does the cache actually return real data, or are we hitting the fallback?
     Fallback would mask whatever is in the daily_spread_cache.

  C. CROSS-PATH COMPARISON — for the SAME fold 1 trade list, compare:
       flat  cost per trade (30 bps × notional, both entry + exit)
       dynamic cost per trade (Week 5 spread + impact)
     If dynamic < flat by a plausible margin → improvement is genuine.
     If dynamic = 0 or dynamic > flat → likely bug.
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from engine_daily.cost_engine import (
    load_cost_data, compute_pair_trade_cost, assign_kappa_tier,
    _lookup_daily_spread, _FALLBACK_HALF_SPREAD_BPS, _FALLBACK_SPREAD_STD,
)


def section(n: int, name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  STEP {n} — {name}")
    print(f"{'='*70}")


def step_A_unit_sanity() -> bool:
    """Verify cost primitives produce expected numbers on hand-checked inputs."""
    section("A", "Unit sanity on cost_engine primitives")

    # ---- Kappa tier ----
    assert assign_kappa_tier(5.0) == 0.3,  f"Kappa tier wrong for 5 bps: {assign_kappa_tier(5.0)}"
    assert assign_kappa_tier(8.0) == 0.5,  f"Kappa tier wrong for 8 bps"
    assert assign_kappa_tier(15.0) == 0.5, f"Kappa tier wrong for 15 bps"
    assert assign_kappa_tier(20.0) == 0.5, f"Kappa tier wrong for 20 bps"
    assert assign_kappa_tier(20.1) == 0.8, f"Kappa tier wrong for 20.1 bps"
    assert assign_kappa_tier(50.0) == 0.8, f"Kappa tier wrong for 50 bps"
    print(f"  [PASS] kappa tiers: 5->0.3, 15->0.5, 25->0.8 (boundaries 8, 20)")

    # ---- compute_pair_trade_cost on synthetic input ----
    # Build a minimal CostData by hand
    from engine_daily.cost_engine import CostData
    idx = pd.MultiIndex.from_tuples([
        ("AAA", pd.Timestamp("2024-01-15")),
        ("BBB", pd.Timestamp("2024-01-15")),
        ("AAA", pd.Timestamp("2024-01-30")),
        ("BBB", pd.Timestamp("2024-01-30")),
    ], names=["ticker", "date"])
    daily_spread = pd.DataFrame({
        "half_spread_bps": [5.0, 5.0, 5.0, 5.0],   # 5 bps half-spread each
        "spread_std_1d":   [2.0, 2.0, 2.0, 2.0],   # 2 bps std
    }, index=idx)
    cd = CostData(daily_spread=daily_spread,
                  kappa_map={"AAA": 0.5, "BBB": 0.5}, fallback_kappa=0.5)

    # Hand calculation:
    #   notional per leg = 10000
    #   entry spread cost:  notional_per_leg * (5 + 5) / 10000 = $10
    #   entry impact cost:  notional_per_leg * (0.5*2 + 0.5*2) / 10000 = $2
    #   exit spread + impact same as entry
    #   round-trip spread+impact = $24
    #   borrow: short leg = $10k, holding = 15 calendar days, rate = 50 bps/yr
    #     borrow = (50/10000)/365 * 10000 * 15 = $2.054
    #   total ≈ $26
    cost = compute_pair_trade_cost(
        cd, "AAA", "BBB",
        entry_date=pd.Timestamp("2024-01-15"),
        exit_date=pd.Timestamp("2024-01-30"),
        notional_per_leg=10_000.0,
        side_a=+1,
    )
    print(f"  Synthetic trade: notional=$10k/leg, half_spread=5bps, std=2bps, 15 cal days")
    print(f"     spread_$={cost['spread_$']:.2f}  (expect ~$20: 4 sides x leg x 5bps x $10k / 10000)")
    print(f"     impact_$={cost['impact_$']:.2f}  (expect ~$4: 4 sides x leg x 1bps x $10k / 10000)")
    print(f"     borrow_$={cost['borrow_$']:.2f}  (expect ~$2.05)")
    print(f"     total_$ ={cost['total_$']:.2f}  (expect ~$26)")
    assert 19.0 < cost["spread_$"] < 21.0, f"spread mag wrong: {cost['spread_$']}"
    assert 3.0 < cost["impact_$"] < 5.0,   f"impact mag wrong: {cost['impact_$']}"
    assert 1.5 < cost["borrow_$"] < 2.5,   f"borrow mag wrong: {cost['borrow_$']}"
    print(f"  [PASS] cost magnitudes within expected hand-calc bounds")

    # ---- Edge case: same-day trade -> borrow = 0 ----
    cost_same_day = compute_pair_trade_cost(
        cd, "AAA", "BBB",
        entry_date=pd.Timestamp("2024-01-15"),
        exit_date=pd.Timestamp("2024-01-15"),
        notional_per_leg=10_000.0,
        side_a=+1,
    )
    assert cost_same_day["borrow_$"] == 0.0, "Same-day trade should have zero borrow"
    print(f"  [PASS] same-day trade -> borrow $0 correctly")

    return True


def step_B_lookup_hit_rate(cd) -> dict:
    """Measure what % of (ticker, date) lookups return real data vs fallback."""
    section("B", "Lookup hit rate on real fold-1 data")

    # Simulate what a fold-1 run would do: for each (ticker, date) the engine
    # would query, count cache hits vs fallback misses.
    from run_v4_pipeline import _load_all_daily, _slice_daily, DATA_DAILY
    from engine_daily import discovery_daily

    print("  Loading daily cache + running discovery to get fold-1 tickers/dates...")
    cache = _load_all_daily(DATA_DAILY)
    trading = _slice_daily(cache, "2023-01-01", "2023-01-31")
    formation = _slice_daily(cache, "2022-01-01", "2022-12-31")

    pairs_df, factor_state = discovery_daily.run(formation, hl_max=30.0, hl_min=5.0)
    tickers_in_fold = sorted({t for pair in zip(pairs_df["ticker_a"], pairs_df["ticker_b"])
                              for t in pair})
    trading_dates = sorted({d for tk in trading for d in trading[tk].index})

    print(f"  Fold 1 has {len(tickers_in_fold)} unique tickers, {len(trading_dates)} trading days")
    print(f"  Total potential lookups: {len(tickers_in_fold) * len(trading_dates):,}")

    hit_real = 0
    hit_fallback = 0
    sample_lookups = []
    for tk in tickers_in_fold[:200]:   # sample 200 tickers
        for date in trading_dates:
            hs, sd = _lookup_daily_spread(cd, tk, date)
            is_fallback = (hs == _FALLBACK_HALF_SPREAD_BPS and sd == _FALLBACK_SPREAD_STD)
            if is_fallback:
                hit_fallback += 1
            else:
                hit_real += 1
            if len(sample_lookups) < 5:
                sample_lookups.append((tk, date.date(), hs, sd, "FALLBACK" if is_fallback else "REAL"))

    total = hit_real + hit_fallback
    pct_real = 100.0 * hit_real / max(total, 1)
    print(f"\n  Sample lookups:")
    for tk, d, hs, sd, status in sample_lookups:
        print(f"    {tk:<6s} {d}  half_spread={hs:.2f} bps  spread_std={sd:.2f} bps  [{status}]")
    print(f"\n  Hit summary: {hit_real:,} real ({pct_real:.1f}%), "
          f"{hit_fallback:,} fallback ({100-pct_real:.1f}%)")

    if pct_real < 80:
        print(f"  [WARN] Less than 80% real hits — fallback is masking actual cost!")
    elif pct_real > 95:
        print(f"  [PASS] >95% real hits — lookup is hitting cache properly")
    else:
        print(f"  [OK] {pct_real:.0f}% real hits — acceptable but watch for edge cases")

    return {"pct_real": pct_real, "n_lookups": total,
            "tickers": tickers_in_fold, "dates": trading_dates}


def step_C_cross_path(cd, fold1_meta) -> None:
    """Compare per-trade cost: flat vs dynamic on real fold 1 trades."""
    section("C", "Cross-path comparison: flat vs dynamic on fold 1 trades")

    from run_v4_pipeline import _load_all_daily, _slice_daily, DATA_DAILY
    from engine_daily import discovery_daily
    from engine_daily.engine_daily import run_fold_daily
    from engine.phase1_cointegration.factor_residual import project_residual

    cache = _load_all_daily(DATA_DAILY)
    formation = _slice_daily(cache, "2022-01-01", "2022-12-31")
    trading = _slice_daily(cache, "2023-01-01", "2023-01-31")
    pairs_df, factor_state = discovery_daily.run(formation, hl_max=30.0, hl_min=5.0)

    W = factor_state["loadings_W"]
    fact_tk = factor_state["tickers"]
    resid_form = factor_state["residual_log_prices"]
    resid_trade_dict = project_residual(trading, W, fact_tk, min_obs=10)
    for tk in list(resid_trade_dict.keys()):
        if tk not in resid_form.columns:
            continue
        s_form_tk = resid_form[tk].dropna()
        if len(s_form_tk) and len(resid_trade_dict[tk]):
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
    resid_trade = pd.concat(resid_trade_dict, axis=1)

    # Run TWICE — once flat, once dynamic
    print("  Running fold 1 with FLAT cost...")
    res_flat = run_fold_daily(pairs_df, resid_form, resid_trade,
                              entry_z=2.0, hard_sl_z=4.0, z_window=60,
                              cost_data=None)
    print("  Running fold 1 with DYNAMIC cost...")
    res_dyn = run_fold_daily(pairs_df, resid_form, resid_trade,
                             entry_z=2.0, hard_sl_z=4.0, z_window=60,
                             cost_data=cd)

    print(f"\n  Pair results: flat={len(res_flat)}, dynamic={len(res_dyn)}")

    # Verify gross P&L is IDENTICAL (only costs should differ)
    gross_flat_sum = sum(df["daily_pnl_gross"].sum() for df in res_flat.values())
    gross_dyn_sum  = sum(df["daily_pnl_gross"].sum() for df in res_dyn.values())
    print(f"\n  Gross P&L: flat=${gross_flat_sum:>+12,.2f}  dynamic=${gross_dyn_sum:>+12,.2f}")
    assert abs(gross_flat_sum - gross_dyn_sum) < 1e-6, \
        f"BUG: gross P&L should be IDENTICAL across cost modes (only net differs)"
    print(f"  [PASS] gross P&L identical (good — only net differs)")

    # Now compare per-trade costs
    flat_costs, dyn_costs = [], []
    notionals = []
    for key in set(res_flat.keys()) & set(res_dyn.keys()):
        df_f, df_d = res_flat[key], res_dyn[key]
        # Sum entry+exit+borrow per pair
        cost_f = (df_f["cost_entry"] + df_f["cost_exit"] + df_f["borrow_cost"]).sum()
        cost_d = (df_d["cost_entry"] + df_d["cost_exit"] + df_d["borrow_cost"]).sum()
        # Number of trades = entries
        pos_f = df_f["position"].values
        n_tr = int(((pos_f != 0) & (np.r_[0, pos_f[:-1]] == 0)).sum())
        if n_tr > 0:
            flat_costs.append(cost_f / n_tr)
            dyn_costs.append(cost_d / n_tr)
            notionals.append(float(df_f.attrs.get("notional", 0.0)))

    flat_arr = np.array(flat_costs)
    dyn_arr = np.array(dyn_costs)
    notional_arr = np.array(notionals)

    print(f"\n  Per-trade cost ($) across {len(flat_arr)} pairs that traded:")
    print(f"    flat    : median ${np.median(flat_arr):.2f}, mean ${flat_arr.mean():.2f}")
    print(f"    dynamic : median ${np.median(dyn_arr):.2f}, mean ${dyn_arr.mean():.2f}")
    print(f"    ratio dyn/flat: median {np.median(dyn_arr/flat_arr):.3f}, mean {(dyn_arr/flat_arr).mean():.3f}")

    print(f"\n  Per-trade cost as bps of FULL pair notional:")
    flat_bps = 10000 * flat_arr / notional_arr
    dyn_bps = 10000 * dyn_arr / notional_arr
    print(f"    flat    : median {np.median(flat_bps):.1f} bps  (expected ~60bps = 30bps x 2 sides)")
    print(f"    dynamic : median {np.median(dyn_bps):.1f} bps  (expected: depends on real spreads)")

    # Sanity check: are dynamic costs SHOCKINGLY low (e.g., near zero)?
    if np.median(dyn_arr) < 1.0:
        print(f"\n  [WARN] dynamic cost median < $1 per trade — suspect bug in calc!")
    elif np.median(dyn_arr) < np.median(flat_arr) * 0.1:
        print(f"\n  [WARN] dynamic cost is < 10% of flat cost — suspiciously low!")
    elif np.median(dyn_arr) > np.median(flat_arr):
        print(f"\n  [PASS] dynamic cost is HIGHER than flat (more conservative)")
    else:
        ratio = np.median(dyn_arr) / np.median(flat_arr)
        print(f"\n  [INFO] dynamic cost is {ratio*100:.0f}% of flat — moderate reduction")

    # Sample 5 trades and print component breakdown
    print(f"\n  Sample 5 trade-level cost components (dynamic):")
    sample_keys = list(set(res_dyn.keys()))[:5]
    for ta, tb in sample_keys:
        df_d = res_dyn[(ta, tb)]
        n_entries = int(((df_d["position"].values != 0) &
                         (np.r_[0, df_d["position"].values[:-1]] == 0)).sum())
        ce = df_d["cost_entry"].sum()
        cx = df_d["cost_exit"].sum()
        cb = df_d["borrow_cost"].sum()
        notional = df_d.attrs.get("notional", 0.0)
        if n_entries:
            print(f"    {ta}/{tb}: notional=${notional:,.0f}, "
                  f"{n_entries} trades, "
                  f"entry=${ce:.2f} exit=${cx:.2f} borrow=${cb:.2f}  "
                  f"(per trade: ${(ce+cx+cb)/n_entries:.2f})")


def main() -> int:
    print("V4 cost-engine audit — Week 5 dynamic vs V4 baseline flat")

    # STEP A: unit sanity
    ok = step_A_unit_sanity()

    # Load cost data (needed for Step B and C)
    print("\nLoading cost data...")
    cd = load_cost_data()
    print(f"  cache: {len(cd.daily_spread):,} rows, kappa map: {len(cd.kappa_map)} tickers")

    # STEP B: lookup hit rate
    meta = step_B_lookup_hit_rate(cd)

    # STEP C: cross-path comparison
    step_C_cross_path(cd, meta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
