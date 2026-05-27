"""Live trading loop entry point. Run as: python live/main.py"""
from __future__ import annotations

# CRITICAL: Force single-threaded BLAS BEFORE any numpy/scipy/sklearn import.
# Backtest set this in scripts/run_v4_pipeline.py for determinism; live must
# match or Z-scores will drift by ~1e-10 between runs (small, but a violation
# of bit-identity invariant in week6_live_invariants.md).
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def configure_logging() -> None:
    log_dir = Path("./live/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_dir / "live_{time:YYYYMMDD}.log", rotation="00:00", retention="30 days",
               level="DEBUG")


async def run() -> None:
    """Main event loop. Phase 2+ implementation."""
    logger.info("live engine starting")
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    asyncio.run(run())
