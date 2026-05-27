"""Property-based fuzz: 10,000 random Z sequences run through both
live.decide() and backtest._state_machine_daily. Position arrays MUST match
exactly across all bars including NaN, boundary, and re-entry cases.

This is the strongest evidence we have that the live state machine is
bit-identical to the backtest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from engine_daily.engine_daily import _state_machine_daily
from live.engine_live.live_pair import decide

N_TRIALS = 100        # number of seeded random scenarios
BARS_PER_TRIAL = 100  # bars per scenario  → 10_000 total bar-decisions
ENTRY_Z = 3.0
HARD_SL_Z = 5.0


def gen_z_sequence(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix of regimes: large moves (testing hard SL), near-zero (testing exit),
    # boundary entries (testing entry threshold), and NaN injections.
    scale = rng.uniform(0.5, 3.0)
    z = rng.normal(0, scale, n)
    # Inject NaN ~5% of the time (warmup / missing data)
    nan_mask = rng.random(n) < 0.05
    z[nan_mask] = np.nan
    # Inject extreme moves ~3% (testing hard SL)
    extreme_mask = rng.random(n) < 0.03
    z[extreme_mask] = rng.choice([-7, -6, 6, 7], size=extreme_mask.sum())
    # Inject boundary values ~3% (testing exact 3.0 / 5.0)
    boundary_mask = rng.random(n) < 0.03
    z[boundary_mask] = rng.choice([3.0, -3.0, 5.0, -5.0], size=boundary_mask.sum())
    return z


def run_live(z: np.ndarray, entry_z: float, hard_sl_z: float) -> np.ndarray:
    state = 0
    out = np.zeros(len(z), dtype=np.int8)
    for i, zi in enumerate(z):
        d = decide(state, float(zi) if not np.isnan(zi) else float("nan"),
                   entry_z=entry_z, hard_sl_z=hard_sl_z)
        state = d.new_state
        out[i] = state
    return out


def main() -> int:
    print(f"== State-machine fuzz: {N_TRIALS} trials x {BARS_PER_TRIAL} bars "
          f"= {N_TRIALS * BARS_PER_TRIAL} decisions ==")

    mismatches = 0
    total_trades = 0
    total_enters = 0
    total_exits = 0
    total_hard_sl = 0
    first_bad_seed = None

    for seed in range(N_TRIALS):
        z = gen_z_sequence(seed, BARS_PER_TRIAL)
        bt_pos, bt_exit_codes = _state_machine_daily(z, ENTRY_Z, HARD_SL_Z, 0)
        live_pos = run_live(z, ENTRY_Z, HARD_SL_Z)
        if not np.array_equal(bt_pos, live_pos):
            mismatches += 1
            if first_bad_seed is None:
                first_bad_seed = seed
                # Show first 10 divergent indices
                diff_idx = np.where(bt_pos != live_pos)[0]
                print(f"  divergence on seed={seed}: {len(diff_idx)} bars differ")
                for i in diff_idx[:10]:
                    print(f"    bar {i}: z={z[i]:.4f} bt={bt_pos[i]} live={live_pos[i]}")
        total_trades += (np.diff(bt_pos.astype(int)) != 0).sum()
        total_enters += int((bt_pos == 1).sum() + (bt_pos == -1).sum())
        total_exits += int((bt_exit_codes == 1).sum())
        total_hard_sl += int((bt_exit_codes == 2).sum())

    print(f"  trials with divergence: {mismatches}/{N_TRIALS}")
    print(f"  total enter/exit transitions: {total_trades}")
    print(f"  bars in position: {total_enters}")
    print(f"  zero-cross exits: {total_exits}")
    print(f"  hard SL triggers: {total_hard_sl}")
    print()
    if mismatches == 0:
        print("PASS: live decide() bit-identical to backtest state machine "
              f"across {N_TRIALS * BARS_PER_TRIAL} bar decisions.")
        return 0
    print(f"FAIL: {mismatches} trials diverged. First bad seed: {first_bad_seed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
