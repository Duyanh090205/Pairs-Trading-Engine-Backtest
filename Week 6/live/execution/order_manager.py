"""Idempotent order submission with safety gates.

Submit path (in order):
  1. hardstop.is_tripped()  → REFUSE
  2. persist.is_halted(conn) → REFUSE  (kill_switch DB row)
  3. orders table lookup by client_order_id → if exists, return existing (no double submit)
  4. broker submit → record in orders table → return
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class OrderRequest:
    pair_id: str
    bar_ts: str
    leg: str
    ticker: str
    side: str            # 'buy' | 'sell'
    qty: float
    order_type: str      # 'market' | 'limit' | 'moo'
    limit_price: float | None = None

    def client_order_id(self) -> str:
        # Include order_type + limit_price so a market vs limit order with same
        # logical parameters get DIFFERENT ids (per Day-2 deep-audit Bug #2).
        raw = (f"{self.pair_id}|{self.bar_ts}|{self.leg}|{self.side}|"
               f"{self.qty:.4f}|{self.order_type}|{self.limit_price or ''}")
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class OrderResult:
    client_order_id: str
    broker_order_id: str | None
    status: str
    submitted: bool         # True if this call actually submitted; False if dedupe hit
    refused_reason: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit_order(trading_client, conn: sqlite3.Connection, req: OrderRequest,
                 decision_price: float, dry_run: bool = False) -> OrderResult:
    """Submit `req` idempotently. Returns OrderResult.

    Safety gates (FIRST, before any broker call):
      - hardstop tripped → refuse
      - kill_switch halted → refuse
      - duplicate client_order_id present → return existing
    """
    from live.safety import hardstop
    from live.state.persist import is_halted, log_event

    coid = req.client_order_id()

    # Gate 1: hardstop
    if hardstop.is_tripped():
        log_event(conn, "order_refused", "ERROR",
                  f"hardstop tripped — refusing {req.pair_id} {req.ticker} {req.side}",
                  {"client_order_id": coid})
        return OrderResult(coid, None, "refused_hardstop", False, "hardstop_tripped")

    # Gate 2: kill_switch DB halt
    halted, halt_reason = is_halted(conn)
    if halted:
        log_event(conn, "order_refused", "ERROR",
                  f"kill_switch halted ({halt_reason}) — refusing {req.pair_id}",
                  {"client_order_id": coid})
        return OrderResult(coid, None, "refused_halt", False, f"halted:{halt_reason}")

    # Gate 3: idempotency
    row = conn.execute(
        "SELECT broker_order_id, status FROM orders WHERE client_order_id = ?",
        (coid,),
    ).fetchone()
    if row is not None:
        log_event(conn, "order_dedupe", "INFO",
                  f"client_order_id already present ({req.ticker}); returning existing",
                  {"client_order_id": coid})
        return OrderResult(coid, row["broker_order_id"], row["status"], False, None)

    # Broker submit
    broker_id: str | None = None
    raw_resp: dict = {}
    if not dry_run:
        try:
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            side_enum = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
            if req.order_type == "limit":
                ord_req = LimitOrderRequest(
                    symbol=req.ticker, qty=req.qty, side=side_enum,
                    time_in_force=TimeInForce.DAY,
                    limit_price=req.limit_price,
                    client_order_id=coid,
                )
            else:  # market | moo
                tif = TimeInForce.OPG if req.order_type == "moo" else TimeInForce.DAY
                ord_req = MarketOrderRequest(
                    symbol=req.ticker, qty=req.qty, side=side_enum,
                    time_in_force=tif, client_order_id=coid,
                )
            resp = trading_client.submit_order(ord_req)
            broker_id = str(getattr(resp, "id", "") or "")
            raw_resp = {"status": str(getattr(resp, "status", "")), "id": broker_id}
        except Exception as e:
            log_event(conn, "order_error", "ERROR",
                      f"broker submit failed: {type(e).__name__}: {e}",
                      {"client_order_id": coid})
            raw_resp = {"error": f"{type(e).__name__}: {e}"}

    status = raw_resp.get("status", "submitted") if "error" not in raw_resp else "rejected"
    conn.execute(
        "INSERT INTO orders (client_order_id, broker_order_id, pair_id, bar_ts, ticker, "
        "side, qty, order_type, limit_price, status, submitted_ts, decision_price, raw_response) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (coid, broker_id, req.pair_id, req.bar_ts, req.ticker, req.side, req.qty,
         req.order_type, req.limit_price, status, _now(),
         decision_price, json.dumps(raw_resp)),
    )
    log_event(conn, "order_submit", "INFO",
              f"{req.ticker} {req.side} {req.qty} @{req.order_type} → {status}",
              {"client_order_id": coid, "broker_order_id": broker_id})
    return OrderResult(coid, broker_id, status, True, None)
