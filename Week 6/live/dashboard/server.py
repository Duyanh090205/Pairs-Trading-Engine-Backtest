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


@app.get("/api/engine_info")
def engine_info():
    """Runtime info about the engine (uptime, pairs loaded, last bar)."""
    import time as _time
    from datetime import datetime, timezone
    info = {
        "engine_running": False,
        "uptime_seconds": 0,
        "pairs_loaded": 0,
        "universe_size": 0,
        "last_bar_ts": None,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from live.main import get_engine
        engine = get_engine()
        if engine is not None:
            info["engine_running"] = True
            info["pairs_loaded"] = len(engine.pair_contexts)
            info["universe_size"] = len(engine.universe_tickers)
            if engine.latest_bar_ts:
                # Most recent bar timestamp across all tickers
                info["last_bar_ts"] = max(engine.latest_bar_ts.values())
            if engine._session_start_equity is not None:
                info["session_start_equity"] = engine._session_start_equity
    except Exception:
        pass
    return info


@app.post("/api/admin/force_decision_now")
async def force_decision_now(request: Request):
    """One-shot manual trigger for daily_decision (paper-only emergency lever).

    Use case: after close on a trading day, kick off today's decision IMMEDIATELY
    instead of waiting for the next 21:05 UTC scheduled wake-up. Useful after
    redeploying with new pair config when you don't want to wait a full day cycle.

    Auth: header `X-Admin-Token` must match env ADMIN_TOKEN (not committed).
    """
    from datetime import datetime, timezone

    token_expected = os.environ.get("ADMIN_TOKEN")
    if not token_expected:
        return JSONResponse({"error": "ADMIN_TOKEN not configured"}, status_code=503)
    token_given = request.headers.get("X-Admin-Token", "")
    if token_given != token_expected:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        from live.main import get_engine
        from live.state.persist import connect, set_last_decision_date
        from live.engine_live.trading_calendar import is_trading_day
    except Exception as e:
        return JSONResponse({"error": f"import failed: {e}"}, status_code=500)

    engine = get_engine()
    if engine is None:
        return JSONResponse({"error": "engine not running"}, status_code=503)

    today = datetime.now(timezone.utc).date()
    if not is_trading_day(today):
        return JSONResponse(
            {"error": f"{today.isoformat()} is not a trading day"}, status_code=400
        )

    # Claim today FIRST so the regular loop doesn't race us on next wake-up.
    try:
        with connect(_DB) as conn:
            set_last_decision_date(conn, today.isoformat())
    except Exception as e:
        logger.warning(f"could not pre-set last_decision_date: {e}")

    n_pairs_before = len(engine.pair_contexts)
    try:
        await engine._run_daily_decisions(today)
    except Exception as e:
        logger.error(f"force_decision_now: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"{type(e).__name__}: {e}", "date": today.isoformat()},
            status_code=500,
        )

    return {
        "ok": True,
        "date": today.isoformat(),
        "pairs_considered": n_pairs_before,
        "note": "Orders (if any) will appear in /api/orders within seconds.",
    }


@app.post("/api/admin/recover_after_status_bug")
async def recover_after_status_bug(request: Request):
    """One-shot recovery after the 'OrderStatus.FILLED' vs 'filled' bug.

    Steps:
      1. Normalize all `orders.status` values to schema-canonical lowercase.
      2. Backfill `positions` table from any pair with both legs filled but no
         open position row.
      3. Clear the kill_switch halt (which was tripped by reconcile mismatch).

    Idempotent — safe to call multiple times.
    Auth via X-Admin-Token header (same as force_decision_now).
    """
    from datetime import datetime, timezone

    token_expected = os.environ.get("ADMIN_TOKEN")
    if not token_expected:
        return JSONResponse({"error": "ADMIN_TOKEN not configured"}, status_code=503)
    if request.headers.get("X-Admin-Token", "") != token_expected:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        from live.execution.order_manager import normalize_status
        from live.state.persist import clear_halt, log_event
    except Exception as e:
        return JSONResponse({"error": f"import failed: {e}"}, status_code=500)

    summary: dict = {"updated_rows": 0, "positions_inserted": 0, "halt_cleared": False}
    try:
        with _conn() as conn:
            # 1. Migrate existing status strings
            rows = conn.execute(
                "SELECT client_order_id, status FROM orders"
            ).fetchall()
            for r in rows:
                norm = normalize_status(r["status"])
                if norm != r["status"]:
                    conn.execute(
                        "UPDATE orders SET status = ? WHERE client_order_id = ?",
                        (norm, r["client_order_id"]),
                    )
                    summary["updated_rows"] += 1

            # 2. Backfill positions for pair_id + bar_ts groups where both legs
            #    are filled but no open position exists.
            filled_groups = conn.execute(
                "SELECT pair_id, bar_ts, COUNT(*) AS n_legs "
                "FROM orders WHERE status = 'filled' "
                "GROUP BY pair_id, bar_ts HAVING COUNT(*) >= 2"
            ).fetchall()
            for g in filled_groups:
                exists = conn.execute(
                    "SELECT 1 FROM positions WHERE pair_id = ? AND exit_ts IS NULL",
                    (g["pair_id"],),
                ).fetchone()
                if exists:
                    continue
                legs = conn.execute(
                    "SELECT ticker, side, qty, fill_price FROM orders "
                    "WHERE pair_id = ? AND bar_ts = ? AND status = 'filled'",
                    (g["pair_id"], g["bar_ts"]),
                ).fetchall()
                if len(legs) < 2:
                    continue
                # Identify ticker_a / ticker_b from pair_id
                if "_" in g["pair_id"]:
                    ta_name, tb_name = g["pair_id"].split("_", 1)
                else:
                    ta_name, tb_name = legs[0]["ticker"], legs[-1]["ticker"]
                leg_a = next((l for l in legs if l["ticker"] == ta_name), legs[0])
                leg_b = next((l for l in legs if l["ticker"] == tb_name), legs[-1])
                direction = 1 if leg_a["side"] == "buy" else -1
                notional_a = float(leg_a["qty"]) * float(leg_a["fill_price"] or 0)
                notional_b = float(leg_b["qty"]) * float(leg_b["fill_price"] or 0)
                beta = notional_b / notional_a if notional_a > 0 else 1.0
                conn.execute(
                    "INSERT OR IGNORE INTO positions "
                    "(pair_id, side_a, side_b, beta, direction, notional_a, notional_b, "
                    " entry_ts, entry_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (g["pair_id"], ta_name, tb_name, beta, direction,
                     notional_a, notional_b,
                     datetime.now(timezone.utc).isoformat(), 0.0),
                )
                summary["positions_inserted"] += 1
                log_event(conn, "position_backfilled", "INFO",
                          f"{g['pair_id']} direction={direction} "
                          f"notional_a=${notional_a:.0f} notional_b=${notional_b:.0f}",
                          {"beta_proxy": beta})

            # 3. Clear halt
            halt_row = conn.execute(
                "SELECT halted, reason FROM kill_switch WHERE id = 1"
            ).fetchone()
            if halt_row and halt_row["halted"]:
                clear_halt(conn)
                log_event(conn, "halt_cleared", "WARN",
                          "kill_switch halt cleared via admin endpoint after "
                          f"status-normalization bug fix (was: {halt_row['reason']})")
                summary["halt_cleared"] = True
    except Exception as e:
        logger.error(f"recover_after_status_bug: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"{type(e).__name__}: {e}", "summary": summary},
            status_code=500,
        )

    return {"ok": True, "summary": summary}


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
        "fill_qty, fill_price, decision_price, entry_z, submitted_ts, filled_ts "
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
