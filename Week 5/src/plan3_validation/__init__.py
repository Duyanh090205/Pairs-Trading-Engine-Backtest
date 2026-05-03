"""
Plan 3: Validation & Report
Analytics modules + red-flag triggers + 12-section report builder.
"""

from .red_flags import (
    check_dynamic_cost_blowup,
    check_cost_exceeds_alpha,
    check_spread_data_gap,
    check_kappa_instability,
    check_nc_leak,
    check_dsr_degradation,
    check_math_violation,
)
from .sharpe_net import generate_before_after_table
from .cost_waterfall import generate_cost_waterfall
from .regime_costs import analyze_regime_costs, REGIME_MAP
from .impact_validation import validate_impact_prediction
from .kill_zone import analyze_kill_zone
from .negative_control import run_dynamic_negative_control
from .overfitting_net import compute_overfitting_diagnostics
from .sensitivity_oat import run_oat_sensitivity
from .report_builder import assemble_report


__all__ = [
    "check_dynamic_cost_blowup",
    "check_cost_exceeds_alpha",
    "check_spread_data_gap",
    "check_kappa_instability",
    "check_nc_leak",
    "check_dsr_degradation",
    "check_math_violation",
    "generate_before_after_table",
    "generate_cost_waterfall",
    "analyze_regime_costs",
    "REGIME_MAP",
    "validate_impact_prediction",
    "analyze_kill_zone",
    "run_dynamic_negative_control",
    "compute_overfitting_diagnostics",
    "run_oat_sensitivity",
    "assemble_report",
]
