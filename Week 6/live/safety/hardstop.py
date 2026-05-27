"""Account-level HARDSTOP — non-overridable circuit breaker.

Distinct from monitor/kill_switch.py:
  - kill_switch: auto-halt with operator-resumable behavior (clears via UI/CLI)
  - hardstop:    final brake. Cancels all orders, flattens positions, refuses
                 new trades. Cleared only by removing a flag file + restart.

Triggers (any one trips it):
  1. Account equity dropped >= max_equity_drop_pct from session start
  2. Cumulative realized loss >= max_realized_loss_usd
  3. Manual flag file present at HARDSTOP_FLAG_PATH
  4. >= max_consecutive_errors errors from broker within rolling window
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HARDSTOP_FLAG_PATH = Path(os.environ.get("HARDSTOP_FLAG_PATH", "./live/state/HARDSTOP.flag"))


@dataclass(frozen=True)
class HardstopLimits:
    max_equity_drop_pct: float = 0.05         # 5% — well below backtest max DD
    max_realized_loss_usd: float = 5_000.0    # $5k on $100k paper = 5%
    max_consecutive_errors: int = 20          # broker errors in 5-min window
    error_window_seconds: int = 300


@dataclass
class HardstopState:
    tripped: bool
    reason: str | None
    triggered_ts: str | None


def check(conn: sqlite3.Connection, session_start_equity: float,
          current_equity: float, limits: HardstopLimits | None = None) -> HardstopState:
    """Evaluate all hardstop triggers. Returns state. If tripped: caller MUST halt."""
    limits = limits or HardstopLimits()

    if HARDSTOP_FLAG_PATH.exists():
        return _trip(conn, "manual_flag_present")

    if session_start_equity > 0:
        drop_pct = (session_start_equity - current_equity) / session_start_equity
        if drop_pct >= limits.max_equity_drop_pct:
            return _trip(conn, f"equity_drop_{drop_pct:.4f}_ge_{limits.max_equity_drop_pct}")

    realized_loss = _cumulative_realized_loss(conn)
    if realized_loss >= limits.max_realized_loss_usd:
        return _trip(conn, f"realized_loss_${realized_loss:.2f}_ge_${limits.max_realized_loss_usd}")

    err_count = _recent_error_count(conn, limits.error_window_seconds)
    if err_count >= limits.max_consecutive_errors:
        return _trip(conn, f"broker_errors_{err_count}_ge_{limits.max_consecutive_errors}_in_{limits.error_window_seconds}s")

    return HardstopState(tripped=False, reason=None, triggered_ts=None)


def _cumulative_realized_loss(conn: sqlite3.Connection) -> float:
    """Sum of negative realized_pnl on closed positions."""
    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM positions "
        "WHERE exit_ts IS NOT NULL AND realized_pnl < 0"
    ).fetchone()
    return -float(row["pnl"]) if row["pnl"] is not None else 0.0


def _recent_error_count(conn: sqlite3.Connection, window_s: int) -> int:
    cutoff = datetime.now(timezone.utc).timestamp() - window_s
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log "
        "WHERE level IN ('ERROR', 'CRITICAL') AND ts >= ?",
        (cutoff_iso,),
    ).fetchone()
    return int(row["n"])


def _trip(conn: sqlite3.Connection, reason: str) -> HardstopState:
    """Trip the hardstop. Uses persist.log_event for audit (single write path)."""
    from live.state.persist import log_event   # local import: avoid circular
    ts = datetime.now(timezone.utc).isoformat()
    HARDSTOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HARDSTOP_FLAG_PATH.write_text(f"{ts}\n{reason}\n")
    log_event(conn, "hardstop_tripped", "CRITICAL", reason,
              payload={"flag_path": str(HARDSTOP_FLAG_PATH)})
    return HardstopState(tripped=True, reason=reason, triggered_ts=ts)


def is_tripped() -> bool:
    """Cheap file-only check — call before every order submission."""
    return HARDSTOP_FLAG_PATH.exists()


def clear(operator_note: str) -> None:
    """Manually clear hardstop. Requires operator note (logged for audit)."""
    if not HARDSTOP_FLAG_PATH.exists():
        return
    HARDSTOP_FLAG_PATH.unlink()
    log_dir = HARDSTOP_FLAG_PATH.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "hardstop_clear_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{operator_note}\n")
