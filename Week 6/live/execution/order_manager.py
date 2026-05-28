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

    def client_order_id(self, attempt: int = 0) -> str:
        # Include order_type + limit_price so a market vs limit order with same
        # logical parameters get DIFFERENT ids (per Day-2 deep-audit Bug #2).
        # attempt>0 distinguishes retries after a previous attempt was canceled
        # or rejected, so the new submission gets a fresh client_order_id and
        # the broker (Alpaca) doesn't see a duplicate coid.
        raw = (f"{self.pair_id}|{self.bar_ts}|{self.leg}|{self.side}|"
               f"{self.qty:.4f}|{self.order_type}|{self.limit_price or ''}")
        if attempt > 0:
            raw += f"|attempt={attempt}"
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


def normalize_status(s: str | None) -> str:
    """Normalize broker status strings to schema-canonical lowercase values.

    Alpaca returns 'OrderStatus.FILLED' / 'OrderStatus.ACCEPTED' / etc. Schema
    expects 'filled' / 'submitted' / etc. All downstream comparisons depend on
    this canonical form — normalize at every write site.
    """
    if not s:
        return ""
    low = s.lower()
    if low.startswith("orderstatus."):
        low = low[len("orderstatus."):]
    # Map Alpaca-only vocabulary to schema vocabulary
    if low == "partially_filled":
        return "partial"
    if low == "accepted":
        return "submitted"
    if low == "done_for_day":
        return "expired"
    if low in ("pending_new", "pending_replace", "replaced", "stopped",
               "suspended", "calculated"):
        return "submitted"
    if low == "pending_cancel":
        return "canceled"
    return low


# Statuses that mean a previous submission attempt is DEAD — safe to re-submit
# under a new client_order_id. Anything not in this set is treated as still
# alive (or successfully filled) and dedupe blocks re-submission.
_DEAD_STATUSES = frozenset({
    "canceled", "orderstatus.canceled",
    "rejected", "orderstatus.rejected",
    "expired", "orderstatus.expired",
    "refused_hardstop", "refused_halt",
})


def _is_dead_status(status: str | None) -> bool:
    return (status or "").lower() in _DEAD_STATUSES


def submit_order(trading_client, conn: sqlite3.Connection, req: OrderRequest,
                 decision_price: float, dry_run: bool = False) -> OrderResult:
    """Submit `req` idempotently. Returns OrderResult.

    Safety gates (FIRST, before any broker call):
      - hardstop tripped → refuse
      - kill_switch halted → refuse
      - duplicate client_order_id present and ALIVE → return existing
      - duplicate client_order_id but DEAD (canceled/rejected/expired) →
        walk attempt counter until we find a fresh coid, then submit
    """
    from live.safety import hardstop
    from live.state.persist import is_halted, log_event

    # Gate 1: hardstop
    if hardstop.is_tripped():
        coid_for_log = req.client_order_id()
        log_event(conn, "order_refused", "ERROR",
                  f"hardstop tripped — refusing {req.pair_id} {req.ticker} {req.side}",
                  {"client_order_id": coid_for_log})
        return OrderResult(coid_for_log, None, "refused_hardstop", False, "hardstop_tripped")

    # Gate 2: kill_switch DB halt
    halted, halt_reason = is_halted(conn)
    if halted:
        coid_for_log = req.client_order_id()
        log_event(conn, "order_refused", "ERROR",
                  f"kill_switch halted ({halt_reason}) — refusing {req.pair_id}",
                  {"client_order_id": coid_for_log})
        return OrderResult(coid_for_log, None, "refused_halt", False, f"halted:{halt_reason}")

    # Gate 3: idempotency with retry walk
    attempt = 0
    coid = req.client_order_id(attempt=0)
    while True:
        row = conn.execute(
            "SELECT broker_order_id, status FROM orders WHERE client_order_id = ?",
            (coid,),
        ).fetchone()
        if row is None:
            break    # fresh coid — proceed to submit
        if not _is_dead_status(row["status"]):
            # Existing order is still alive / filled — return it (true dedupe)
            log_event(conn, "order_dedupe", "INFO",
                      f"client_order_id already present ({req.ticker}); "
                      f"status={row['status']}, returning existing",
                      {"client_order_id": coid})
            return OrderResult(coid, row["broker_order_id"], row["status"], False, None)
        # Existing row is dead — advance attempt counter and try again
        attempt += 1
        if attempt > 50:    # safety cap; in practice 1–2 attempts is plenty
            log_event(conn, "order_error", "ERROR",
                      f"too many dead retries ({req.ticker}); giving up",
                      {"client_order_id": coid})
            return OrderResult(coid, None, "rejected", False, "too_many_retries")
        coid = req.client_order_id(attempt=attempt)
        log_event(conn, "order_resubmit", "INFO",
                  f"previous attempt {attempt-1} was {row['status']}; "
                  f"retrying {req.ticker} with attempt={attempt}",
                  {"client_order_id": coid, "previous_status": row["status"]})

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

    raw_status = raw_resp.get("status", "submitted") if "error" not in raw_resp else "rejected"
    status = normalize_status(raw_status)
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
