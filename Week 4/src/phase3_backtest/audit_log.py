"""
Phase 3 — Per-Fold Audit Log

Writes a 7-section .txt audit log per fold to results/logs/.

Sections:
  [1] Parameter hash (fold config)
  [2] Kalman delta + multi-criterion metrics
  [3] Trade log sample (first 20 trades)
  [4] Comparative metrics (primary vs NC)
  [5] Timestamp verification proof (exec_ts > signal_ts for 100%)
  [6] NC bootstrap results
  [7] Red flag status + environment hash

Red flag triggers (from spec):
  Lookahead leak   : Sharpe > 5.0
  NC discrimination: primary Sharpe <= NC bootstrap threshold
  Kalman degenerate: var(prior)/R outside [0.05, 4.0]
  Delta boundary   : auto-selected delta at grid edge
  Delta instability: delta jumps >2 steps vs previous fold
  Universal fail   : no delta passed constraints
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_DELTA_GRID      = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
_SHARPE_LOOKAHEAD_THRESHOLD = 5.0
_LOG_DIR_DEFAULT = "results/logs"


def write_audit_log(
    fold_n: int,
    fold_metrics: dict,
    nc_metrics: dict | None,
    latency_results: dict | None,
    delta: float | None,
    delta_metrics: dict,
    config: dict,
    prev_delta: float | None = None,
    output_dir: str = _LOG_DIR_DEFAULT,
) -> str:
    """
    Write 7-section audit log for one fold.

    Parameters
    ----------
    fold_n         : fold number (1-based)
    fold_metrics   : output of metrics_runner.run_fold_pnl
    nc_metrics     : output of neg_control.run_neg_control (or None)
    latency_results: output of latency.run_latency_sweep (or None)
    delta          : selected Kalman delta (None if universal fail)
    delta_metrics  : {delta_val: {metric_kurt, median_HL, median_ACF78}}
    config         : fold config dict (formation/trading window, params)
    prev_delta     : delta from previous fold (for instability check)
    output_dir     : directory to write fold_NN_audit.txt

    Returns
    -------
    str path to written audit log
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"fold_{fold_n:02d}_audit.txt")

    # --- Compute red flags ---
    sharpe        = fold_metrics.get("sharpe", 0.0)
    lookahead_ok  = fold_metrics.get("lookahead_ok", True)
    kalman_deg    = fold_metrics.get("kalman_degenerate", False)

    flag_lookahead   = (not lookahead_ok) or (sharpe > _SHARPE_LOOKAHEAD_THRESHOLD)
    flag_nc_fail     = False
    flag_kalman_deg  = kalman_deg
    flag_delta_bound = _check_delta_boundary(delta)
    flag_delta_instab = _check_delta_instability(delta, prev_delta)
    flag_univ_fail   = (delta is None)

    if nc_metrics is not None:
        flag_nc_fail = not nc_metrics.get("nc_pass", True)

    config_hash = _hash_config(config)

    lines = []

    # -------------------------------------------------------------------------
    lines += [
        f"=== FOLD {fold_n:02d} AUDIT LOG ===",
        f"Generated: {_now_str()}",
        "",
        "[1] PARAMETER HASH",
        f"config_hash   = {config_hash}",
    ]
    for k, v in config.items():
        lines.append(f"  {k:<24} = {v}")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[2] KALMAN DELTA + MULTI-CRITERION METRICS"]
    if delta is None:
        lines.append("  Selected delta: NONE (universal constraint fail)")
    else:
        lines.append(f"  Selected delta: {delta:.0e}")
        if flag_delta_bound:
            lines.append("  WARNING: delta at grid boundary")
        if flag_delta_instab:
            lines.append(f"  WARNING: delta instability (prev={prev_delta:.0e})")

    lines.append("  Grid results:")
    for d in _DELTA_GRID:
        if d in delta_metrics:
            m = delta_metrics[d]
            kurt  = m.get("metric_kurt",  float("nan"))
            hl    = m.get("median_HL",    float("nan"))
            acf78 = m.get("median_ACF78", float("nan"))
            sel   = " <-- SELECTED" if d == delta else ""
            feasible = (
                1.0 <= hl <= 10.0 and acf78 > 0.7
                and not (np.isnan(hl) or np.isnan(acf78))
            )
            feas_tag = " [FEASIBLE]" if feasible else ""
            lines.append(
                f"    delta={d:.0e}  kurt_dev={kurt:6.3f}  "
                f"HL={hl:6.2f}d  ACF78={acf78:.3f}{feas_tag}{sel}"
            )
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[3] TRADE LOG SAMPLE (first 20 trades)"]
    trade_log = fold_metrics.get("trade_log", [])
    if not trade_log:
        lines.append("  No trades this fold.")
    else:
        header = f"  {'entry_ts':<24} {'exit_ts':<24} {'dir':>4} {'bars':>6} {'gross_bps':>10} {'net_bps':>10} {'exit_reason'}"
        lines.append(header)
        for t in trade_log[:20]:
            lines.append(
                f"  {str(t.get('entry_ts','')):<24} "
                f"{str(t.get('exit_ts','')):<24} "
                f"{t.get('direction', 0):>4} "
                f"{t.get('n_bars', 0):>6} "
                f"{t.get('gross_bps', 0.0):>10.1f} "
                f"{t.get('net_bps', 0.0):>10.1f} "
                f"{t.get('exit_reason', '?')}"
            )
        if len(trade_log) > 20:
            lines.append(f"  ... ({len(trade_log) - 20} more trades not shown)")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[4] COMPARATIVE METRICS"]
    cost = fold_metrics.get("cost_decomp", {})
    lines += [
        f"  Primary Sharpe  : {sharpe:.4f}",
        f"  MaxDD (bar)     : {fold_metrics.get('max_dd', 0.0):.4f}",
        f"  CAGR            : {fold_metrics.get('cagr', 0.0):.4f}",
        f"  Calmar          : {fold_metrics.get('calmar', 0.0):.4f}",
        f"  Win Rate        : {fold_metrics.get('win_rate', 0.0):.2%}",
        f"  N Trades        : {fold_metrics.get('n_trades', 0)}",
        f"  Avg Hold (bars) : {fold_metrics.get('avg_holding_bars', 0.0):.1f}",
        f"  Avg Net (bps)   : {fold_metrics.get('avg_net_bps', 0.0):.2f}",
        f"  Cost commission : ${cost.get('commission', 0.0):,.2f}",
        f"  Cost borrow     : ${cost.get('borrow', 0.0):,.2f}",
        f"  Cost rebalance  : ${cost.get('rebalance', 0.0):,.2f}",
    ]

    if nc_metrics is not None:
        lines += [
            f"  NC empirical Sharpe  : {nc_metrics.get('empirical_sharpe', 'N/A')}",
            f"  NC synthetic Sharpe  : {nc_metrics.get('synthetic_sharpe', 0.0):.4f}",
            f"  NC bootstrap mean+2sd: {nc_metrics.get('bootstrap_threshold', 0.0):.4f}",
            f"  NC PASS              : {nc_metrics.get('nc_pass', False)}",
            f"  NC source            : {nc_metrics.get('nc_source', 'unknown')}",
        ]

    if latency_results is not None:
        lines.append("  Latency sweep:")
        for row in latency_results.get("latency_table", []):
            lines.append(
                f"    {row['lag']:>6}  Sharpe={row['sharpe']:.4f}"
            )
        lines.append(f"  Latency pass: {latency_results.get('latency_pass', False)}")

    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[5] TIMESTAMP VERIFICATION PROOF"]
    n_trades = fold_metrics.get("n_trades", 0)
    la_ok    = fold_metrics.get("lookahead_ok", True)
    lines += [
        f"  Total trades              : {n_trades}",
        f"  exec_ts > signal_ts (100%): {la_ok}",
        f"  LOOKAHEAD STATUS          : {'PASS' if la_ok else 'FAIL'}",
    ]
    if sharpe > _SHARPE_LOOKAHEAD_THRESHOLD:
        lines.append(f"  WARNING: Sharpe={sharpe:.2f} > 5.0 — investigate for lookahead")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[6] NEGATIVE CONTROL BOOTSTRAP"]
    if nc_metrics is None:
        lines.append("  NC bootstrap not run this fold.")
    else:
        b_mean   = nc_metrics.get("bootstrap_mean", 0.0)
        b_std    = nc_metrics.get("bootstrap_std", 0.0)
        thresh   = nc_metrics.get("bootstrap_threshold", 0.0)
        nc_pass  = nc_metrics.get("nc_pass", False)
        pct_pos  = _pct_above(nc_metrics.get("bootstrap_sharpes", np.array([])), 0.0)
        lines += [
            f"  Bootstrap Sharpes (N=1000): mean={b_mean:.4f}  std={b_std:.4f}",
            f"  Threshold (mean+2sd)       : {thresh:.4f}",
            f"  Primary Sharpe             : {sharpe:.4f}",
            f"  Bootstrap pct > 0          : {pct_pos:.1%}",
            f"  NC PASS                    : {nc_pass}",
            f"  Discrimination STATUS      : {'PASS' if nc_pass else 'FAIL'}",
        ]
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[7] RED FLAG STATUS + ENVIRONMENT HASH"]
    flags = {
        "Lookahead leak (Sharpe>5 or pos mismatch)": flag_lookahead,
        "NC discrimination fail":                    flag_nc_fail,
        "Kalman spread degenerate":                  flag_kalman_deg,
        "Delta at grid boundary":                    flag_delta_bound,
        "Delta instability (>2 steps)":              flag_delta_instab,
        "Universal constraint fail (no delta)":      flag_univ_fail,
    }
    any_critical = flag_lookahead or flag_nc_fail or flag_univ_fail
    for name, val in flags.items():
        tag = "ERROR" if val and name in (
            "Lookahead leak (Sharpe>5 or pos mismatch)",
            "NC discrimination fail",
            "Universal constraint fail (no delta)",
        ) else ("WARNING" if val else "OK")
        lines.append(f"  [{tag:<7}] {name}: {val}")

    lines += [
        "",
        f"  OVERALL STATUS: {'CRITICAL' if any_critical else 'OK'}",
        "",
        "  Environment:",
        f"    Python   : {sys.version.split()[0]}",
        f"    Platform : {platform.platform()}",
    ]
    try:
        import numpy, pandas, numba, statsmodels
        lines += [
            f"    numpy    : {numpy.__version__}",
            f"    pandas   : {pandas.__version__}",
            f"    numba    : {numba.__version__}",
            f"    statsmodels: {statsmodels.__version__}",
        ]
    except ImportError:
        pass
    lines.append(f"    env_hash : {_env_hash()}")
    lines += ["", "=== END AUDIT LOG ==="]

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log.info("Audit log written: %s", path)

    # Log any critical flags
    if flag_lookahead:
        log.error("RED FLAG: Lookahead leak detected — Fold %d HALT", fold_n)
    if flag_nc_fail:
        log.error("RED FLAG: NC discrimination fail — Fold %d", fold_n)
    if flag_univ_fail:
        log.error("RED FLAG: Universal delta constraint fail — Fold %d", fold_n)

    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_delta_boundary(delta: float | None) -> bool:
    if delta is None:
        return False
    return delta in (_DELTA_GRID[0], _DELTA_GRID[-1])


def _check_delta_instability(delta: float | None, prev_delta: float | None) -> bool:
    if delta is None or prev_delta is None:
        return False
    try:
        curr_idx = _DELTA_GRID.index(delta)
        prev_idx = _DELTA_GRID.index(prev_delta)
        return abs(curr_idx - prev_idx) > 2
    except ValueError:
        return False


def _hash_config(config: dict) -> str:
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _env_hash() -> str:
    env_str = f"{sys.version}{platform.platform()}"
    try:
        import numpy, pandas, numba
        env_str += f"{numpy.__version__}{pandas.__version__}{numba.__version__}"
    except ImportError:
        pass
    return hashlib.sha256(env_str.encode()).hexdigest()[:12]


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pct_above(arr: np.ndarray, threshold: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr > threshold))
