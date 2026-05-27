"""End-of-month / end-of-run flatten: close any positions still open.

When called:
  - At last trading day 15:55 ET (just before close) for monthly cycle
  - At end of paper run for cleanup before reporting

For each open position:
  1. Determine exit side per leg (reverse of entry direction)
  2. Submit closing orders via order_manager (idempotent client_order_id)
  3. trade_journal.record_exit fires once both legs fill (via TradingStream)

Hardstop integration: if hardstop is tripped, force-flatten is STILL ATTEMPTED
(unlike new-entry orders). Reason: when manual halt triggers, operator wants
positions CLOSED, not held indefinitely. submit_order's hardstop gate must be
bypassed for closing orders.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from live.execution.order_manager import OrderRequest, OrderResult


@dataclass
class FlattenAction:
    pair_id: str
    leg_a_order: OrderResult
    leg_b_order: OrderResult


def _next_bar_ts(now_utc_iso: str) -> str:
    """Bar timestamp for the closing decision. Use current second-precise UTC
    string so the close orders for THIS flatten event are grouped by bar_ts."""
    return now_utc_iso


def flatten_all_open_positions(trading_client, conn: sqlite3.Connection,
                               bar_ts: str,
                               bypass_hardstop: bool = True) -> list[FlattenAction]:
    """Close all currently-open pair positions.

    For each open position (entry recorded in `positions` table):
      direction +1 → SELL ticker_a, BUY ticker_b (cover short)
      direction -1 → BUY ticker_a (cover), SELL ticker_b
    """
    open_positions = conn.execute(
        "SELECT pair_id, side_a, side_b, beta, direction, notional_a, notional_b, "
        "entry_ts FROM positions WHERE exit_ts IS NULL"
    ).fetchall()
    if not open_positions:
        return []

    # Best-effort: get last fill prices to estimate qty for the closing legs.
    # In production, the engine should know per-leg qty from the fill history.
    out: list[FlattenAction] = []
    for pos in open_positions:
        pair_id = pos["pair_id"]
        ta, tb = pos["side_a"], pos["side_b"]
        direction = int(pos["direction"])
        qty_a, qty_b = _reconstruct_leg_qty(conn, pair_id, ta, tb)
        if qty_a is None or qty_b is None or qty_a <= 0 or qty_b <= 0:
            from live.state.persist import log_event
            log_event(conn, "eom_flatten_skip", "WARN",
                      f"{pair_id}: cannot reconstruct leg quantities for flatten")
            continue
        side_a = "sell" if direction == 1 else "buy"
        side_b = "buy" if direction == 1 else "sell"
        req_a = OrderRequest(
            pair_id=pair_id, bar_ts=bar_ts, leg="A_close", ticker=ta,
            side=side_a, qty=qty_a, order_type="market",
        )
        req_b = OrderRequest(
            pair_id=pair_id, bar_ts=bar_ts, leg="B_close", ticker=tb,
            side=side_b, qty=qty_b, order_type="market",
        )
        if bypass_hardstop:
            out_a = _submit_force(trading_client, conn, req_a)
            out_b = _submit_force(trading_client, conn, req_b)
        else:
            from live.execution.order_manager import submit_order
            out_a = submit_order(trading_client, conn, req_a, decision_price=0.0)
            out_b = submit_order(trading_client, conn, req_b, decision_price=0.0)
        out.append(FlattenAction(pair_id=pair_id, leg_a_order=out_a, leg_b_order=out_b))
    return out


def _reconstruct_leg_qty(conn: sqlite3.Connection, pair_id: str,
                          ticker_a: str, ticker_b: str) -> tuple[float | None, float | None]:
    """Find the qty of each leg from the entry fills."""
    row_a = conn.execute(
        "SELECT SUM(fill_qty) AS q FROM orders "
        "WHERE pair_id = ? AND ticker = ? AND status = 'filled' "
        "AND side IN ('buy', 'sell')",
        (pair_id, ticker_a),
    ).fetchone()
    row_b = conn.execute(
        "SELECT SUM(fill_qty) AS q FROM orders "
        "WHERE pair_id = ? AND ticker = ? AND status = 'filled' "
        "AND side IN ('buy', 'sell')",
        (pair_id, ticker_b),
    ).fetchone()
    qa = float(row_a["q"]) if row_a and row_a["q"] else None
    qb = float(row_b["q"]) if row_b and row_b["q"] else None
    return qa, qb


def _submit_force(trading_client, conn: sqlite3.Connection,
                  req: OrderRequest) -> OrderResult:
    """Submit a closing order, BYPASSING hardstop + kill_switch gates.

    Closing positions during an emergency halt is SAFER than holding them.
    Idempotency (client_order_id dedupe) is still enforced.
    """
    import json
    from datetime import datetime, timezone
    from live.state.persist import log_event

    coid = req.client_order_id()
    # Idempotency check (only gate that still applies)
    row = conn.execute(
        "SELECT broker_order_id, status FROM orders WHERE client_order_id = ?", (coid,),
    ).fetchone()
    if row is not None:
        log_event(conn, "order_dedupe", "INFO",
                  f"close coid {coid} already present; skipping")
        return OrderResult(coid, row["broker_order_id"], row["status"], False, None)

    broker_id: str | None = None
    raw_resp: dict = {}
    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        side_enum = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        ord_req = MarketOrderRequest(
            symbol=req.ticker, qty=req.qty, side=side_enum,
            time_in_force=TimeInForce.DAY, client_order_id=coid,
        )
        resp = trading_client.submit_order(ord_req)
        broker_id = str(getattr(resp, "id", "") or "")
        raw_resp = {"status": str(getattr(resp, "status", "")), "id": broker_id}
    except Exception as e:
        log_event(conn, "eom_flatten_error", "ERROR",
                  f"close submit failed for {req.ticker}: {type(e).__name__}: {e}")
        raw_resp = {"error": f"{type(e).__name__}: {e}"}

    status = raw_resp.get("status", "submitted") if "error" not in raw_resp else "rejected"
    conn.execute(
        "INSERT INTO orders (client_order_id, broker_order_id, pair_id, bar_ts, ticker, "
        "side, qty, order_type, status, submitted_ts, decision_price, raw_response) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (coid, broker_id, req.pair_id, req.bar_ts, req.ticker, req.side, req.qty,
         req.order_type, status,
         datetime.now(timezone.utc).isoformat(), 0.0, json.dumps(raw_resp)),
    )
    log_event(conn, "eom_flatten_submit", "INFO",
              f"close {req.ticker} {req.side} {req.qty} -> {status}")
    return OrderResult(coid, broker_id, status, True, None)
