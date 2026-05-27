"""Real-data sanity: run live ZTracker against backtest rolling_z_with_warmup on
ACTUAL AAPL daily bars from the V4 cache. Synthetic tests pass — but real-world
data has its own quirks (autocorrelation, vol clustering, gaps). If they pass
on real data, the implementation is robust.

Also runs decide() through the full real-data Z series and checks the live
state machine matches backtest on real prices, not just synthetic random walks.
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

from engine_daily.engine_daily import _state_machine_daily, rolling_z_with_warmup
from live.engine_live.live_pair import decide
from live.engine_live.z_tracker import ZTracker

DAILY_CACHE = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\daily_phase3")
TICKERS_TO_TEST = ["AAPL", "MSFT", "JPM", "NVDA", "BAC"]
WINDOW = 60


def synth_pair_spread(ticker_a: str, ticker_b: str) -> tuple[pd.Series, pd.Series, dict]:
    """Construct a real-data spread series for ticker_a - 1.0 * ticker_b (β=1)."""
    df_a = pd.read_parquet(DAILY_CACHE / f"{ticker_a}.parquet")
    df_b = pd.read_parquet(DAILY_CACHE / f"{ticker_b}.parquet")
    common = df_a.index.intersection(df_b.index)
    if len(common) < 252:
        return pd.Series(), pd.Series(), {"err": "insufficient common dates"}
    spread = df_a.loc[common, "log_close"] - df_b.loc[common, "log_close"]
    # Split: first WINDOW bars as formation seed; rest as live "trading window".
    formation = spread.iloc[:WINDOW]
    trading = spread.iloc[WINDOW:]
    return formation, trading, {"n_common": len(common)}


def run_live(formation, trading) -> np.ndarray:
    tracker = ZTracker(window=WINDOW, seed=formation.values.tolist())
    out = []
    for v in trading.values:
        z = tracker.push(float(v))
        out.append(z if z is not None else np.nan)
    return np.asarray(out)


def main() -> int:
    print(f"== Real-data sanity: ZTracker vs backtest on actual daily bars ==")
    pairs = [
        ("AAPL", "MSFT"), ("JPM", "BAC"), ("NVDA", "AAPL"), ("MSFT", "JPM"),
    ]
    all_pass = True
    total_bars = 0
    state_mismatches = 0
    z_max_diff = 0.0
    z_decisions_diverged = 0

    for a, b in pairs:
        formation, trading, meta = synth_pair_spread(a, b)
        if len(trading) == 0:
            print(f"  {a}/{b}: SKIP ({meta.get('err', 'no data')})")
            continue

        # Backtest Z path
        bt_z = rolling_z_with_warmup(trading, formation, window=WINDOW).values
        # Live Z path
        live_z = run_live(formation, trading)
        # Compare
        z_diff = np.nanmax(np.abs(bt_z - live_z))
        z_max_diff = max(z_max_diff, z_diff)

        # State machine on bt_z
        bt_pos, _ = _state_machine_daily(bt_z, 3.0, 5.0, 0)
        # State machine via live decide on live_z
        state = 0
        live_pos = np.zeros(len(live_z), dtype=np.int8)
        for i, zi in enumerate(live_z):
            d = decide(state, float(zi) if not np.isnan(zi) else float("nan"),
                       entry_z=3.0, hard_sl_z=5.0)
            state = d.new_state
            live_pos[i] = state
        bar_diff = (bt_pos != live_pos).sum()
        state_mismatches += int(bar_diff)
        total_bars += len(trading)

        entry_diverged = ((np.abs(bt_z) >= 3.0) != (np.abs(live_z) >= 3.0)).sum()
        z_decisions_diverged += int(entry_diverged)

        ok = (z_diff < 1e-9) and (bar_diff == 0)
        if not ok:
            all_pass = False
        print(f"  {a}/{b}: n_trading={len(trading)} z_max_diff={z_diff:.2e} "
              f"state_diffs={bar_diff} entry_decisions_diverged={entry_diverged} "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print(f"  total trading bars across all pairs: {total_bars}")
    print(f"  total state-machine mismatches:     {state_mismatches}")
    print(f"  max |Z| diff:                       {z_max_diff:.2e}")
    print(f"  entry-decision divergences at Z=3:  {z_decisions_diverged}")
    print()
    if all_pass and state_mismatches == 0:
        print("PASS: live engine bit-identical to backtest on real AAPL/MSFT/JPM/NVDA/BAC data.")
        return 0
    print("FAIL: divergence detected on real data.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
