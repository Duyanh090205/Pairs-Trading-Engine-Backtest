"""Cross-path test: live ZTracker.push vs backtest rolling_z_with_warmup.

Both compute "rolling 60-day Z on residual spread with formation-window warmup."
If they disagree, live engine will silently trade at different Z thresholds than
backtest-verified shipping config (Z=3.0). That's a deep-audit-bug wiring issue.
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
import pandas as pd

from engine_daily.engine_daily import rolling_z_with_warmup
from live.engine_live.z_tracker import ZTracker


def synthesize_input(seed: int = 0, formation_n: int = 120, trading_n: int = 40,
                     window: int = 60):
    rng = np.random.default_rng(seed)
    formation_idx = pd.date_range("2024-01-01", periods=formation_n, freq="B")
    trading_idx = pd.date_range(formation_idx[-1] + pd.tseries.offsets.BDay(),
                                periods=trading_n, freq="B")
    formation = pd.Series(rng.normal(0, 1, formation_n).cumsum() * 0.01,
                          index=formation_idx)
    trading = pd.Series(rng.normal(0, 1, trading_n).cumsum() * 0.01 + formation.iloc[-1],
                        index=trading_idx)
    return formation, trading, window


def run_backtest(formation: pd.Series, trading: pd.Series, window: int) -> np.ndarray:
    """Backtest path. Returns the live-window Z values."""
    z = rolling_z_with_warmup(trading, formation, window=window)
    return z.values


def run_live(formation: pd.Series, trading: pd.Series, window: int) -> np.ndarray:
    """Live path: seed ZTracker with last `window` formation bars, then push trading bars."""
    seed = formation.tail(window).tolist()
    tracker = ZTracker(window=window, seed=seed)
    out = []
    for v in trading.values:
        z = tracker.push(float(v))
        out.append(z if z is not None else np.nan)
    return np.asarray(out)


def report_divergence(z_bt: np.ndarray, z_live: np.ndarray) -> None:
    diff = z_bt - z_live
    abs_diff = np.abs(diff)
    print(f"  backtest  Z[:5] = {z_bt[:5]}")
    print(f"  live      Z[:5] = {z_live[:5]}")
    print(f"  max |diff|      = {abs_diff.max():.6g}")
    print(f"  ratio bt/live   = {np.nanmean(z_bt / z_live):.6g} "
          f"(theoretical pstdev/std with n=60: {np.sqrt(60/59):.6g})")
    print(f"  any sign flip?  = {np.any(np.sign(z_bt) != np.sign(z_live))}")
    # Show where entry threshold (|Z|>=3) decisions diverge
    bt_entry = np.abs(z_bt) >= 3.0
    live_entry = np.abs(z_live) >= 3.0
    divergent_bars = (bt_entry != live_entry).sum()
    print(f"  bars where bt-vs-live entry decision diverges at Z=3.0: {divergent_bars}/{len(z_bt)}")


def main() -> int:
    formation, trading, window = synthesize_input(seed=0)
    z_bt = run_backtest(formation, trading, window)
    z_live = run_live(formation, trading, window)

    print("== ZTracker vs rolling_z_with_warmup — cross-path comparison ==")
    report_divergence(z_bt, z_live)

    tol = 1e-9
    max_diff = np.nanmax(np.abs(z_bt - z_live))
    if max_diff < tol:
        print(f"PASS: max diff {max_diff} < tol {tol}. No divergence.")
        return 0
    else:
        print(f"FAIL: max diff {max_diff} >= tol {tol}. Live ZTracker diverges from backtest.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
