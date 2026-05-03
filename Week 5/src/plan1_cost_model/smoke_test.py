import pandas as pd
import numpy as np
import os
import sys

WEEK5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(WEEK5_ROOT)

from src.plan1_cost_model import (
    calculate_spread_cost,
    assign_kappa_tier,
    calculate_impact_cost,
    calculate_borrow_cost,
    calculate_round_trip_cost,
    calculate_static_round_trip_cost,
    validate_cost_log,
)

def run_plan1_smoke_test():
    print("--- Running Plan 1 Smoke Test (Synthetic Data) ---")

    np.random.seed(42)

    # ------------------------------------------------------------------ #
    # 1. Test assign_kappa_tier directly
    # ------------------------------------------------------------------ #
    print("\n--- Kappa Tier Assignment ---")
    assert assign_kappa_tier(4.7) == 0.3,  "SPY (~4.7 bps) should be Tier 1, kappa=0.3"
    assert assign_kappa_tier(7.99) == 0.3, "Just below 8 bps boundary should be Tier 1"
    assert assign_kappa_tier(8.0) == 0.5,  "Exactly 8.0 bps should be Tier 2, kappa=0.5"
    assert assign_kappa_tier(10.0) == 0.5, "10 bps should be Tier 2, kappa=0.5"
    assert assign_kappa_tier(20.0) == 0.5, "Exactly 20.0 bps should be Tier 2, kappa=0.5"
    assert assign_kappa_tier(32.0) == 0.8, "VTRS (~32 bps) should be Tier 3, kappa=0.8"
    print("[PASS] assign_kappa_tier: all tier boundaries correct (Tier1<8, Tier2 8-20, Tier3>20).")

    # ------------------------------------------------------------------ #
    # 2. Test sensitivity multiplier produces monotonically ordered costs
    # ------------------------------------------------------------------ #
    print("\n--- Sensitivity Grid Monotonicity ---")
    test_notional = 100_000.0
    test_sigma = 5.0
    kappa = assign_kappa_tier(10.0)   # Tier 2, kappa=0.5
    cost_0_5x = calculate_impact_cost(kappa, test_sigma, test_notional, multiplier=0.5)
    cost_1_0x = calculate_impact_cost(kappa, test_sigma, test_notional, multiplier=1.0)
    cost_1_5x = calculate_impact_cost(kappa, test_sigma, test_notional, multiplier=1.5)
    assert cost_0_5x < cost_1_0x < cost_1_5x, (
        f"Sensitivity grid not monotonic: {cost_0_5x:.4f} < {cost_1_0x:.4f} < {cost_1_5x:.4f}"
    )
    print(f"[PASS] Sensitivity grid monotonic: "
          f"0.5x=${cost_0_5x:.2f} < 1.0x=${cost_1_0x:.2f} < 1.5x=${cost_1_5x:.2f}")

    # ------------------------------------------------------------------ #
    # 3. Three cost regimes produce distinct, correctly ordered totals
    # ------------------------------------------------------------------ #
    print("\n--- Three Cost Regimes ---")
    notional_A = notional_B = 100_000.0
    half_spread = 5.0    # bps
    sigma = 3.0          # bps (spread_std_1d)
    kappa_dyn = assign_kappa_tier(10.0)   # 0.5

    # Gross: zero cost
    gross_cost = 0.0

    # Static 60 bps: 30 bps/leg on each of A and B, entry + exit
    static_cost = calculate_static_round_trip_cost(notional_A, notional_B, notional_A, notional_B, tc_bps_per_leg=30.0)

    # Dynamic: spread + impact on each leg, entry + exit, no borrow (intraday)
    entry_A = calculate_spread_cost(half_spread, notional_A) + calculate_impact_cost(kappa_dyn, sigma, notional_A)
    entry_B = calculate_spread_cost(half_spread, notional_B) + calculate_impact_cost(kappa_dyn, sigma, notional_B)
    dynamic_cost = calculate_round_trip_cost(entry_A, entry_B, entry_A, entry_B, borrow_cost=0.0)

    assert gross_cost == 0.0, "Gross regime must be zero cost"
    assert static_cost > 0.0, "Static regime must be positive"
    assert dynamic_cost > 0.0, "Dynamic regime must be positive"
    assert static_cost != dynamic_cost, "Static and dynamic should produce different costs for variable spreads"
    print(f"[PASS] Three regimes compute without error: "
          f"Gross=${gross_cost:.2f}, Static=${static_cost:.2f}, Dynamic=${dynamic_cost:.2f}")

    # Static cost sanity: 30bps * 4 legs * 100k = $120
    expected_static = 4 * 100_000.0 * (30.0 / 10000.0)
    assert abs(static_cost - expected_static) < 1e-8, \
        f"Static cost formula wrong: got {static_cost:.4f}, expected {expected_static:.4f}"
    print(f"[PASS] Static 60bps formula correct: 4 legs × 30bps × $100k = ${expected_static:.2f}")

    # ------------------------------------------------------------------ #
    # 4. Generate 100 synthetic trades and validate cost_log schema
    # ------------------------------------------------------------------ #
    print("\n--- 100-Trade Cost Log ---")
    n_trades = 100

    days = np.random.randint(1, 10, size=n_trades)
    entry_ts = pd.Timestamp('2022-01-03 10:00:00', tz='US/Eastern') + pd.to_timedelta(days, unit='d')
    is_overnight = np.random.choice([True, False], size=n_trades)
    exit_ts = entry_ts + pd.to_timedelta(np.where(is_overnight, 24, 2), unit='h')

    notionals_A = np.random.uniform(50_000, 150_000, size=n_trades)
    notionals_B = np.random.uniform(50_000, 150_000, size=n_trades)
    half_spreads_A = np.random.uniform(1.0, 5.0, size=n_trades)
    half_spreads_B = np.random.uniform(5.0, 15.0, size=n_trades)
    kappa_A = assign_kappa_tier(3.0)   # Tier 1 (spread ~3 bps for A)
    kappa_B = assign_kappa_tier(25.0)  # Tier 3 (spread ~25 bps for B)
    sigmas_A = np.random.uniform(0.5, 2.0, size=n_trades)
    sigmas_B = np.random.uniform(2.0, 10.0, size=n_trades)

    spread_costs, impact_costs, borrow_costs, total_costs = [], [], [], []

    for i in range(n_trades):
        e_sp_A = calculate_spread_cost(half_spreads_A[i], notionals_A[i])
        e_sp_B = calculate_spread_cost(half_spreads_B[i], notionals_B[i])
        e_im_A = calculate_impact_cost(kappa_A, sigmas_A[i], notionals_A[i])
        e_im_B = calculate_impact_cost(kappa_B, sigmas_B[i], notionals_B[i])

        x_sp_A = calculate_spread_cost(half_spreads_A[i], notionals_A[i])
        x_sp_B = calculate_spread_cost(half_spreads_B[i], notionals_B[i])
        x_im_A = calculate_impact_cost(kappa_A, sigmas_A[i], notionals_A[i])
        x_im_B = calculate_impact_cost(kappa_B, sigmas_B[i], notionals_B[i])

        total_spread = e_sp_A + e_sp_B + x_sp_A + x_sp_B
        total_impact = e_im_A + e_im_B + x_im_A + x_im_B
        borrow = calculate_borrow_cost(notionals_B[i], entry_ts[i], exit_ts[i], 50.0)

        rt = calculate_round_trip_cost(
            e_sp_A + e_im_A, e_sp_B + e_im_B,
            x_sp_A + x_im_A, x_sp_B + x_im_B,
            borrow,
        )
        spread_costs.append(total_spread)
        impact_costs.append(total_impact)
        borrow_costs.append(borrow)
        total_costs.append(rt)

    cost_log = pd.DataFrame({
        'trade_id':              [f"T{i}" for i in range(n_trades)],
        'spread_cost_dollars':   spread_costs,
        'impact_cost_dollars':   impact_costs,
        'borrow_cost_dollars':   borrow_costs,
        'rebalance_cost_dollars': 0.0,
        'total_cost_dollars':    total_costs,
        'net_pnl_dollars':       np.random.uniform(-500, 500, n_trades) - np.array(total_costs),
        'net_return':            np.random.uniform(-0.02, 0.02, n_trades),
    })

    # All costs >= 0
    assert (cost_log['total_cost_dollars'] >= 0).all(), "Negative total costs found"
    print(f"[PASS] All round-trip costs >= 0 (mean=${cost_log['total_cost_dollars'].mean():.2f})")

    # Dollar denomination sanity
    assert cost_log['total_cost_dollars'].mean() > 10.0, "Costs look like bps, not dollars"
    print("[PASS] Costs are dollar-denominated.")

    # Borrow: 0 for intraday, >0 for overnight
    assert (cost_log.loc[~is_overnight, 'borrow_cost_dollars'] == 0).all(), \
        "Borrow cost charged on intraday trades"
    assert (cost_log.loc[is_overnight, 'borrow_cost_dollars'] > 0).all(), \
        "Borrow cost missing on overnight trades"
    print("[PASS] Borrow cost logic: 0 intraday, >0 overnight.")

    # Schema validation
    validate_cost_log(cost_log)
    print("[PASS] Interface contract schema validation passed.")

    print("\nPlan 1 Smoke Test Passed Successfully!")
    return True

if __name__ == "__main__":
    run_plan1_smoke_test()
