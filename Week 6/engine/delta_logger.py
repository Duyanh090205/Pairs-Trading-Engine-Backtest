"""
V2-vs-V3 delta logger
======================

Append one row per pipeline step to results/v3/delta_log.csv so the V3 vs V2
comparison is built incrementally as the pipeline runs (not a one-shot diff
at the end).

Schema:
    timestamp, step, metric, v2_value, v3_value, delta, pct_change, notes
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = "results/v3/delta_log.csv"
_HEADER = ["timestamp", "step", "metric", "v2_value", "v3_value", "delta", "pct_change", "notes"]


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def log_delta(
    step: str,
    metric: str,
    v2_value: Any,
    v3_value: Any,
    notes: str = "",
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> None:
    """
    Append one comparison row. Computes delta and pct_change automatically.

    If v2/v3 are not numeric, delta and pct_change are left blank.
    Creates the CSV with header on first write.
    """
    # ---- HARD STOPS ----
    assert isinstance(step, str) and step, "step must be non-empty string"
    assert isinstance(metric, str) and metric, "metric must be non-empty string"

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v2_f = _safe_float(v2_value)
    v3_f = _safe_float(v3_value)
    delta = pct_change = ""
    if v2_f is not None and v3_f is not None:
        delta = f"{v3_f - v2_f:.6g}"
        if abs(v2_f) > 1e-12:
            pct_change = f"{100.0 * (v3_f - v2_f) / abs(v2_f):.3f}"

    row = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        step,
        metric,
        "" if v2_value is None else str(v2_value),
        "" if v3_value is None else str(v3_value),
        delta,
        pct_change,
        notes,
    ]

    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(_HEADER)
        w.writerow(row)


def reset_log(log_path: str | Path = DEFAULT_LOG_PATH) -> None:
    """Delete and recreate empty log (for fresh runs)."""
    path = Path(log_path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
