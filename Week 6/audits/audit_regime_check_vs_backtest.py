"""Cross-path audit: live regime_check.decide_month vs backtest halt_for_fold.

Per Phase 2 deep-audit Bug B, the live wrapper is supposed to DELEGATE to the
backtest function. If it drifts (date conversion, tz handling, etc.), live halt
decisions will silently diverge from the V4 ship config that gave Sharpe +1.28.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from engine_daily.regime_detector import (
    build_features,
    compute_stress_zscore,
    halt_for_fold,
)
from live.engine_live.regime_check import DAILY_CACHE_DIR, decide_month
from live.state.persist import connect, init_db


def load_minimal_universe() -> list[str]:
    """Load a representative subset for fast comparison."""
    files = sorted(p.stem for p in DAILY_CACHE_DIR.iterdir() if p.suffix == ".parquet")
    return files[:80]


def run_backtest_path(universe: list[str], trade_start: pd.Timestamp) -> tuple[bool, dict]:
    cache = {}
    for tk in universe:
        fp = DAILY_CACHE_DIR / f"{tk}.parquet"
        if fp.exists():
            cache[tk] = pd.read_parquet(fp)
    feats = build_features(cache)
    feats = compute_stress_zscore(feats)
    return halt_for_fold(feats, trade_start)


def run_live_path(universe: list[str], decision_ts: datetime,
                  db_path: Path) -> tuple[bool, dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        dec = decide_month(conn, decision_ts, universe)
    return dec.halted, dec.diag


def main() -> int:
    universe = load_minimal_universe()
    # Pick a month INSIDE the live cache window (May 2025 -> May 2026).
    # 2026-04 has 252 prior trading days in cache for stress_z computation.
    decision_ts = datetime(2026, 4, 15, 16, 0, 0, tzinfo=timezone.utc)
    trade_start = pd.Timestamp("2026-04-01")

    print(f"== regime_check.decide_month vs halt_for_fold ({len(universe)} tickers) ==")
    print(f"  decision_ts: {decision_ts.isoformat()}")
    print(f"  trade_start (backtest): {trade_start}")

    bt_halt, bt_diag = run_backtest_path(universe, trade_start)
    print(f"\n  BACKTEST: halt={bt_halt} stress_z={bt_diag.get('stress_z')} "
          f"q67={bt_diag.get('q67_trailing')}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        live_halt, live_diag = run_live_path(universe, decision_ts, Path(td) / "test.db")
    print(f"  LIVE:     halt={live_halt} stress_z={live_diag.get('stress_z')} "
          f"q67={live_diag.get('q67_trailing')}")

    if bt_halt != live_halt:
        print(f"\nFAIL: halt decisions differ ({bt_halt} vs {live_halt})")
        return 1
    bt_sz = bt_diag.get("stress_z")
    live_sz = live_diag.get("stress_z")
    if bt_sz is None and live_sz is None:
        pass
    elif bt_sz is None or live_sz is None:
        print(f"\nFAIL: stress_z presence differs ({bt_sz} vs {live_sz})")
        return 1
    elif abs(bt_sz - live_sz) > 1e-9:
        print(f"\nFAIL: stress_z differs by {abs(bt_sz - live_sz)}")
        return 1
    bt_q = bt_diag.get("q67_trailing")
    live_q = live_diag.get("q67_trailing")
    if bt_q is not None and live_q is not None and abs(bt_q - live_q) > 1e-9:
        print(f"\nFAIL: q67 differs by {abs(bt_q - live_q)}")
        return 1
    print("\nPASS: live regime_check.decide_month bit-identical to backtest halt_for_fold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
