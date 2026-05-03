"""
Plan 2: Cost Application
Per-trade post-processing pass over Week 4's trade_log/rebalance_log.
Produces data/cost_log.parquet and data/kappa_per_fold.parquet.
"""

from .walk_forward import FOLD_SCHEDULE, FOLD_BY_ID, FoldSpec, calibrate_fold
from .hooks import trade_dynamic_cost, trade_static_cost, rebalance_dynamic_cost
from .orchestrator import (
    run_plan2,
    load_trade_log,
    load_rebalance_log,
    load_spread_lookup,
)

__all__ = [
    "FOLD_SCHEDULE",
    "FOLD_BY_ID",
    "FoldSpec",
    "calibrate_fold",
    "trade_dynamic_cost",
    "trade_static_cost",
    "rebalance_dynamic_cost",
    "run_plan2",
    "load_trade_log",
    "load_rebalance_log",
    "load_spread_lookup",
]
