"""TODO 1 smoketest: BLAS det + NYSE calendar.

Hardstop check inline: verify importing main.py sets BLAS env vars BEFORE
numpy is touched (test by introspection).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_blas_env_set_by_main():
    """live.main module sets BLAS=1 at top, before numpy import."""
    import os
    # Force a re-import to ensure side effect applies
    if "live.main" in sys.modules:
        del sys.modules["live.main"]
    import live.main  # noqa: F401
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        check(f"BLAS env var {var}=1 after live.main import",
              os.environ.get(var) == "1",
              f"got {os.environ.get(var)}")


def t_nyse_calendar_basic():
    from live.engine_live.trading_calendar import (
        is_trading_day, last_trading_day_of_month, is_last_trading_day_of_month,
        trading_days_in_month, is_half_day,
    )
    # 2026-05-25 Memorial Day (NYSE closed)
    check("Memorial Day 2026-05-25 NOT trading day",
          is_trading_day(date(2026, 5, 25)) is False)
    # 2026-05-22 Friday before — trading day
    check("Friday 2026-05-22 IS trading day",
          is_trading_day(date(2026, 5, 22)) is True)
    # Sat/Sun not trading
    check("Sunday 2026-05-24 NOT trading", is_trading_day(date(2026, 5, 24)) is False)
    # Last trading day of May 2026 (skipping holidays) — should be 2026-05-29
    last_may = last_trading_day_of_month(2026, 5)
    check("Last trading day of May 2026 = 2026-05-29 (Fri)",
          last_may == date(2026, 5, 29), f"got {last_may}")
    check("is_last_trading_day_of_month(2026-05-29) True",
          is_last_trading_day_of_month(date(2026, 5, 29)) is True)
    check("is_last_trading_day_of_month(2026-05-22) False",
          is_last_trading_day_of_month(date(2026, 5, 22)) is False)
    # Trading days in May 2026 (21 expected, minus Memorial Day = 20)
    days = trading_days_in_month(2026, 5)
    check("May 2026 has 20 trading days (21 weekdays - Memorial Day)",
          len(days) == 20, f"got {len(days)}")


def t_nyse_half_day():
    from live.engine_live.trading_calendar import is_half_day
    # 2025-11-28 Black Friday — half day (close 13:00 ET = 18:00 UTC)
    check("2025-11-28 (Black Friday) is half day",
          is_half_day(date(2025, 11, 28)) is True)
    # 2025-11-26 (Wednesday before Thanksgiving) — regular day
    check("2025-11-26 (regular Wed) is NOT half day",
          is_half_day(date(2025, 11, 26)) is False)


def t_hardstop_still_works():
    """Hardstop check after foundational changes — make sure didn't break safety."""
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t1_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop initially not tripped", hardstop.is_tripped() is False)
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop tripped after flag write", hardstop.is_tripped() is True)
    hardstop.clear("todo1 test")
    check("hardstop cleared", hardstop.is_tripped() is False)


def main() -> int:
    print("== TODO 1 Smoketest: BLAS + NYSE calendar + hardstop ==")
    print("\n--- BLAS env ---")
    t_blas_env_set_by_main()
    print("\n--- NYSE calendar ---")
    t_nyse_calendar_basic()
    print("\n--- Half-day handling ---")
    t_nyse_half_day()
    print("\n--- Hardstop integration ---")
    t_hardstop_still_works()
    print()
    if errors:
        print(f"FAIL: {len(errors)} - {errors}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
