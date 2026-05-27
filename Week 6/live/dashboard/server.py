"""FastAPI + HTMX dashboard. Server-rendered HTML (no React).

Endpoints per pipeline_week6_live.md §Phase 4:
  GET  /              main dashboard (HTML page)
  GET  /api/status    connectivity strip JSON
  GET  /api/pnl       P&L (gross / net / vs backtest predicted)
  GET  /api/positions open positions list
  GET  /api/orders    recent orders
  GET  /api/drift     realized vs predicted cost (placeholder until Phase 4.25)
  GET  /api/trades    last N trades
  GET  /api/log       last N audit log events
  GET  /api/regime    current regime status
  GET  /api/health    liveness for Render
"""
from __future__ import annotations

import os as _os
# BLAS det — uvicorn doesn't go through main.py; ensure set before numpy imports.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

_BASE = Path(__file__).parent
_DB = Path(os.environ.get("STATE_DB_PATH", "./live/state/live_state.db"))

# Toggle: set ENGINE_ENABLED=true in Render env to auto-start the engine alongside
# the dashboard in the same uvicorn process. Default false for local dashboard-only mode.
_ENGINE_ENABLED = os.environ.get("ENGINE_ENABLED", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Launch engine as concurrent task on dashboard startup (if enabled)."""
    if _ENGINE_ENABLED:
        try:
            from live.main import configure_logging, start_engine_background
            configure_logging()
            await start_engine_background()
            logger.info("Live engine started inside uvicorn process")
        except Exception as e:
            logger.error(f"Engine startup failed: {type(e).__name__}: {e}")
    else:
        logger.info("Dashboard-only mode (ENGINE_ENABLED=false)")
    yield
    # Shutdown: nothing special — uvicorn cancels asyncio tasks


app = FastAPI(title="Week 6 Live Paper Trading Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=str(_BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")


def _conn():
    """Open a Row-factory SQLite connection to the live state DB."""
    import sqlite3
    c = sqlite3.connect(_DB, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


def _safe_query(sql: str, params: tuple = ()) -> list[dict]:
    if not _DB.exists():
        return []
    try:
        with _conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


@app.get("/api/health")
def health():
    return {"ok": True, "db_exists": _DB.exists()}


@app.get("/api/status")
def status():
    """Connectivity strip: kill-switch state, last bar age, broker auth."""
    halted = False
    halt_reason = None
    if _DB.exists():
        with _conn() as conn:
            row = conn.execute(
                "SELECT halted, reason FROM kill_switch WHERE id = 1"
            ).fetchone()
            if row is not None:
                halted = bool(row["halted"])
                halt_reason = row["reason"]
    from live.safety.hardstop import is_tripped
    return {
        "kill_switch_halted": halted,
        "kill_switch_reason": halt_reason,
        "hardstop_tripped": is_tripped(),
        "db_exists": _DB.exists(),
    }


@app.get("/api/positions")
def positions():
    return {"rows": _safe_query(
        "SELECT pair_id, side_a, side_b, beta, direction, notional_a, notional_b, "
        "entry_ts, entry_z FROM positions WHERE exit_ts IS NULL "
        "ORDER BY entry_ts DESC"
    )}


@app.get("/api/orders")
def orders(limit: int = 50):
    return {"rows": _safe_query(
        "SELECT client_order_id, pair_id, ticker, side, qty, order_type, status, "
        "fill_qty, fill_price, decision_price, submitted_ts, filled_ts "
        "FROM orders ORDER BY submitted_ts DESC LIMIT ?",
        (limit,),
    )}


@app.get("/api/trades")
def trades(limit: int = 50):
    """Closed positions (exit_ts NOT NULL) — used as the trade journal."""
    return {"rows": _safe_query(
        "SELECT pair_id, entry_ts, exit_ts, entry_z, exit_z, exit_reason, "
        "realized_pnl, predicted_cost_bps, realized_cost_bps "
        "FROM positions WHERE exit_ts IS NOT NULL "
        "ORDER BY exit_ts DESC LIMIT ?",
        (limit,),
    )}


@app.get("/api/pnl")
def pnl():
    """Realized P&L sums + open-position count."""
    rows = _safe_query(
        "SELECT COALESCE(SUM(realized_pnl), 0) AS realized "
        "FROM positions WHERE exit_ts IS NOT NULL"
    )
    realized = float(rows[0]["realized"]) if rows else 0.0
    open_rows = _safe_query(
        "SELECT COUNT(*) AS n FROM positions WHERE exit_ts IS NULL"
    )
    n_open = int(open_rows[0]["n"]) if open_rows else 0
    return {
        "realized_pnl_usd": realized,
        "open_positions": n_open,
        "backtest_predicted_annual_usd": 700.0,    # ~0.7% on $100k (honest expectation)
        "backtest_sharpe": 1.28,
    }


@app.get("/api/regime")
def regime(limit: int = 12):
    return {"rows": _safe_query(
        "SELECT month, stress_z, q67_thresh, halted, decided_ts "
        "FROM regime_decisions ORDER BY month DESC LIMIT ?",
        (limit,),
    )}


@app.get("/api/log")
def log(limit: int = 50):
    return {"rows": _safe_query(
        "SELECT ts, event, level, message FROM audit_log "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )}


@app.get("/api/drift")
def drift():
    if not _DB.exists():
        return {"n_trades": 0, "mean_drift_bps": None}
    from live.monitor.drift_monitor import compute_drift_bps
    with _conn() as conn:
        return compute_drift_bps(conn, lookback_days=21)


@app.get("/api/comparison")
def comparison():
    """Live vs backtest reference rolling 21-day comparison."""
    if not _DB.exists():
        return {"rows": []}
    from dataclasses import asdict
    from live.monitor.live_vs_backtest import rolling_comparison
    with _conn() as conn:
        rows = rolling_comparison(conn, window_days=21)
    return {"rows": [asdict(r) for r in rows]}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
