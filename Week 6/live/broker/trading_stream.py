"""Alpaca TradingStream — listens for order/fill events and propagates to local state.

Fixes Phase-2 audit Bugs A + E:
  - Updates orders.fill_qty + orders.fill_price + orders.status on fills
  - When both legs of a pair fill, INSERTs into positions table

Separate from market-data WebSocket (which streams bars/trades from data API).
TradingStream uses the trading API URL.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger


@dataclass
class FillEvent:
    client_order_id: str
    broker_order_id: str
    ticker: str
    side: str
    qty: float                    # cumulative filled qty as of this event
    price: float                  # most-recent fill price
    is_final_fill: bool           # True if order fully filled


class TradingStreamHandler:
    """Subscribes to Alpaca trade_updates and applies them to local SQLite."""

    def __init__(self, cfg, conn_factory: Callable[[], sqlite3.Connection]) -> None:
        self.cfg = cfg
        self.conn_factory = conn_factory
        self._stop = False

    async def run(self) -> None:
        from alpaca.trading.stream import TradingStream
        stream = TradingStream(self.cfg.api_key, self.cfg.secret_key, paper=self.cfg.paper)
        stream.subscribe_trade_updates(self._on_event)
        try:
            await stream._run_forever()
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    async def _on_event(self, event) -> None:
        try:
            self.apply_event(event)
        except Exception as e:
            logger.error(f"trade_update handler error: {type(e).__name__}: {e}")

    # ---- Pure logic, testable without a stream ----

    def apply_event(self, event) -> FillEvent | None:
        """Parse a trade_update and persist to local state. Returns FillEvent
        if this was a fill/partial_fill; None for non-fill events (new, canceled, ...)."""
        ev_type = getattr(event, "event", None) or (event.get("event") if isinstance(event, dict) else None)
        order = getattr(event, "order", None) or (event.get("order") if isinstance(event, dict) else None)
        if order is None:
            return None

        def _g(name):
            return getattr(order, name, None) if not isinstance(order, dict) else order.get(name)

        coid = str(_g("client_order_id") or "")
        if not coid:
            return None

        from live.execution.order_manager import normalize_status
        status = normalize_status(str(_g("status") or ""))
        broker_id = str(_g("id") or "")
        filled_qty = float(_g("filled_qty") or 0.0)
        filled_avg_price = float(_g("filled_avg_price") or 0.0)

        conn = self.conn_factory()
        try:
            self._update_order_row(conn, coid, broker_id, status, filled_qty, filled_avg_price, ev_type)
            if ev_type in ("fill", "partial_fill"):
                if filled_qty > 0 and self._is_final_fill(status):
                    self._maybe_open_position(conn, coid)
                    self._maybe_close_position(conn, coid)
                conn.commit()
                return FillEvent(
                    client_order_id=coid, broker_order_id=broker_id,
                    ticker=str(_g("symbol") or ""), side=str(_g("side") or ""),
                    qty=filled_qty, price=filled_avg_price,
                    is_final_fill=self._is_final_fill(status),
                )
            conn.commit()
            return None
        finally:
            conn.close()

    @staticmethod
    def _is_final_fill(status: str) -> bool:
        # `status` here is already passed through normalize_status() in apply_event,
        # but defensive normalize again in case called from a test.
        from live.execution.order_manager import normalize_status
        return normalize_status(status) == "filled"

    _TERMINAL_STATUSES = {"filled", "canceled", "rejected", "expired"}

    def _update_order_row(self, conn, coid, broker_id, status, filled_qty, fill_price, ev_type) -> None:
        """Update orders row but REFUSE backwards transitions.

        Alpaca's TradingStream can replay older events after a reconnect, and
        network reordering can deliver a 'new' event AFTER a 'filled' event for
        the same order. Once an order is in a terminal state (filled/canceled/
        rejected/expired), subsequent updates with non-terminal status MUST be
        ignored — otherwise we'd lose the fill_qty/fill_price and break
        reconcile.
        """
        from live.execution.order_manager import normalize_status
        ts = datetime.now(timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT status FROM orders WHERE client_order_id = ?", (coid,)
        ).fetchone()
        existing_status_norm = normalize_status(existing["status"]) if existing else ""
        status_norm = normalize_status(status)
        if existing is not None and existing_status_norm in self._TERMINAL_STATUSES \
                and status_norm not in self._TERMINAL_STATUSES:
            from live.state.persist import log_event
            log_event(conn, "trade_update_skipped", "WARN",
                      f"ignored out-of-order event: {ev_type} {coid} "
                      f"would move {existing['status']} -> {status}",
                      {"broker_order_id": broker_id})
            return
        filled_ts = ts if status_norm == "filled" else None
        conn.execute(
            "UPDATE orders SET broker_order_id = COALESCE(broker_order_id, ?), "
            "status = ?, fill_qty = ?, fill_price = ?, filled_ts = COALESCE(filled_ts, ?) "
            "WHERE client_order_id = ?",
            (broker_id or None, status, filled_qty or None, fill_price or None,
             filled_ts, coid),
        )
        from live.state.persist import log_event
        log_event(conn, "trade_update", "INFO",
                  f"{ev_type} {coid} status={status} qty={filled_qty} px={fill_price}",
                  {"broker_order_id": broker_id})

    def _maybe_open_position(self, conn, coid: str) -> None:
        """When both legs of a pair are FILLED, INSERT a row in positions.

        Identifies the pair from orders.pair_id. If both legs (A and B) for this
        pair_id+bar_ts are filled and no open position exists yet, opens one.
        Pair entry context (beta, direction, entry_z, notional) is reconstructed
        from the orders + decision_price columns. Direction is +1 if leg A side
        is 'buy', else -1.
        """
        row = conn.execute(
            "SELECT pair_id, bar_ts, side, qty, fill_price, decision_price "
            "FROM orders WHERE client_order_id = ?", (coid,),
        ).fetchone()
        if row is None:
            return
        pair_id = row["pair_id"]
        bar_ts = row["bar_ts"]
        # Look for both legs of this pair at THIS DECISION BAR (not insert-time)
        legs = conn.execute(
            "SELECT pair_id, ticker, side, qty, fill_price, status, bar_ts, entry_z "
            "FROM orders WHERE pair_id = ? AND bar_ts = ?",
            (pair_id, bar_ts),
        ).fetchall()
        if len(legs) < 2:
            return  # waiting for the other leg
        from live.execution.order_manager import normalize_status
        if not all(normalize_status(l["status"]) == "filled" for l in legs):
            return
        # Skip if a position is already open for this pair
        already = conn.execute(
            "SELECT 1 FROM positions WHERE pair_id = ? AND exit_ts IS NULL", (pair_id,),
        ).fetchone()
        if already is not None:
            return
        side_a = pair_id.split("_")[0]
        side_b = pair_id.split("_")[1] if "_" in pair_id else legs[1]["ticker"]
        leg_a = next((l for l in legs if l["ticker"] == side_a), legs[0])
        leg_b = next((l for l in legs if l["ticker"] == side_b), legs[-1])
        direction = 1 if leg_a["side"] == "buy" else -1
        notional_a = float(leg_a["qty"]) * float(leg_a["fill_price"] or 0)
        notional_b = float(leg_b["qty"]) * float(leg_b["fill_price"] or 0)
        # beta proxy from qty ratio (true beta lives in pair_contexts memory)
        beta = notional_b / notional_a if notional_a > 0 else 1.0
        # entry_z written by submit_order from ctx.last_z. Fall back to 0 if a
        # legacy row (pre-migration) lacks it.
        entry_z_vals = [l["entry_z"] for l in legs if l["entry_z"] is not None]
        entry_z = float(entry_z_vals[0]) if entry_z_vals else 0.0
        entry_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO positions "
            "(pair_id, side_a, side_b, beta, direction, notional_a, notional_b, "
            " entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, side_a, side_b, beta, direction, notional_a, notional_b,
             entry_iso, entry_z),
        )
        from live.state.persist import log_event
        log_event(conn, "position_opened", "INFO",
                  f"{pair_id} direction={direction} notional_a=${notional_a:.0f} "
                  f"notional_b=${notional_b:.0f}",
                  {"beta_proxy": beta})
        self._record_predicted_cost(conn, pair_id, side_a, side_b, direction,
                                    notional_a, notional_b, entry_iso)

    def _record_predicted_cost(self, conn, pair_id, side_a, side_b, direction,
                               notional_a, notional_b, entry_iso) -> None:
        """Wire trade_journal.record_entry: predicted_cost_bps from the V4
        dynamic cost model. Only records when the real Week 5 cost cache is
        loadable — the flat 30bp-half-spread fallback would poison the
        drift metric (and the drift_loss kill-switch trigger) with numbers
        ~10x our real spreads. Missing cache -> predicted stays NULL.
        """
        try:
            if not hasattr(self, "_cost_data"):
                from live.monitor.cost_overlay import try_load_cost_data
                self._cost_data = try_load_cost_data()
                if self._cost_data is None:
                    logger.warning("cost cache unavailable — "
                                   "predicted_cost_bps stays NULL")
            if self._cost_data is None:
                return
            from live.monitor.cost_overlay import compute_predicted_cost_usd
            usd = compute_predicted_cost_usd(
                self._cost_data, side_a, side_b, entry_iso, entry_iso,
                notional_per_leg=notional_a, side_a=direction,
            )
            denom = notional_a + notional_b
            if denom <= 0:
                return
            from live.monitor.trade_journal import record_entry
            record_entry(conn, pair_id, usd / denom * 10_000)
        except Exception as e:
            # Telemetry only — never let cost recording break fill handling.
            logger.warning(f"predicted cost recording failed for {pair_id}: "
                           f"{type(e).__name__}: {e}")

    def _maybe_close_position(self, conn, coid: str) -> None:
        """When both EXIT legs of a pair fill, UPDATE positions to set
        exit_ts + realized_pnl. Mirrors _maybe_open_position for the exit side.

        Identifies the open position for this pair. An exit leg's side is the
        REVERSE of the entry leg's side. Walk all filled orders for this pair,
        find the entry pair (earliest 2 fills, opposite sides) and exit pair
        (later 2 fills, opposite sides), compute P&L per leg using direction.
        """
        from live.execution.order_manager import normalize_status
        row = conn.execute(
            "SELECT pair_id FROM orders WHERE client_order_id = ?", (coid,),
        ).fetchone()
        if row is None:
            return
        pair_id = row["pair_id"]

        # Must have an open position to close
        pos = conn.execute(
            "SELECT side_a, side_b, direction, entry_ts, notional_a, notional_b "
            "FROM positions WHERE pair_id = ? AND exit_ts IS NULL",
            (pair_id,),
        ).fetchone()
        if pos is None:
            return

        ta, tb = pos["side_a"], pos["side_b"]
        direction = int(pos["direction"])
        entry_ts = pos["entry_ts"]
        entry_side_a = "buy" if direction == 1 else "sell"
        entry_side_b = "sell" if direction == 1 else "buy"
        exit_side_a = "sell" if direction == 1 else "buy"
        exit_side_b = "buy" if direction == 1 else "sell"

        def _sum_filled(ticker: str, side: str, ts_op: str, ts_ref: str):
            """Sum fills for this leg, filtering by filled_ts relative to
            position.entry_ts. ts_op is '<=' (for entry side, count fills at
            or before entry_ts) or '>' (for exit side, count fills strictly
            after entry_ts so prior lifecycle's exits don't contaminate
            today's just-opened position). Also returns qty-weighted
            decision_price (0.0 when absent — legacy rows) for slippage."""
            row = conn.execute(
                f"SELECT SUM(fill_qty) AS q, "
                f"SUM(fill_qty * fill_price) / NULLIF(SUM(fill_qty), 0) AS px, "
                f"SUM(fill_qty * decision_price) / NULLIF(SUM(fill_qty), 0) AS dpx, "
                f"MAX(filled_ts) AS ts "
                f"FROM orders WHERE pair_id = ? AND ticker = ? "
                f"AND side = ? AND status = 'filled' "
                f"AND filled_ts {ts_op} ?",
                (pair_id, ticker, side, ts_ref),
            ).fetchone()
            return (
                float(row["q"]) if row and row["q"] else 0.0,
                float(row["px"]) if row and row["px"] else 0.0,
                float(row["dpx"]) if row and row["dpx"] else 0.0,
                row["ts"] if row else None,
            )

        # Entry fills: at or before entry_ts (the position's entry timestamp).
        # Exit fills: STRICTLY after entry_ts — this prevents the bug where a
        # prior lifecycle's EOM exits get mixed with today's new entry and
        # falsely close the position immediately.
        eqa, epa, edpa, _ = _sum_filled(ta, entry_side_a, "<=", entry_ts)
        eqb, epb, edpb, _ = _sum_filled(tb, entry_side_b, "<=", entry_ts)
        xqa, xpa, xdpa, xtsa = _sum_filled(ta, exit_side_a, ">", entry_ts)
        xqb, xpb, xdpb, xtsb = _sum_filled(tb, exit_side_b, ">", entry_ts)

        # Both exit legs must be filled
        if xqa <= 0 or xqb <= 0:
            return
        # Entry side must also be there (sanity)
        if eqa <= 0 or eqb <= 0:
            return

        qty_a = min(eqa, xqa)
        qty_b = min(eqb, xqb)
        if direction == 1:
            pnl_a = qty_a * (xpa - epa)         # long A
            pnl_b = qty_b * (epb - xpb)         # short B
        else:
            pnl_a = qty_a * (epa - xpa)         # short A
            pnl_b = qty_b * (xpb - epb)         # long B
        realized_pnl = pnl_a + pnl_b

        # exit_ts = latest of the two exit fills
        exit_ts = max([t for t in (xtsa, xtsb) if t],
                      default=datetime.now(timezone.utc).isoformat())

        # Realized execution cost = broker slippage vs decision price across
        # the 4 legs of THIS lifecycle, normalized to traded notional (bps).
        # Convention (cost_overlay): BUY filled above decision = positive
        # cost; SELL filled below decision = positive cost. Legacy rows with
        # decision_price NULL/0 are excluded from both numerator and
        # denominator (can't measure their slippage).
        slip_usd = 0.0
        traded_usd = 0.0
        legs_slip = (
            (entry_side_a, qty_a, epa, edpa), (entry_side_b, qty_b, epb, edpb),
            (exit_side_a, qty_a, xpa, xdpa), (exit_side_b, qty_b, xpb, xdpb),
        )
        for side, q, fp, dp in legs_slip:
            if q <= 0 or fp <= 0 or dp <= 0:
                continue
            slip_usd += q * ((fp - dp) if side == "buy" else (dp - fp))
            traded_usd += q * dp
        realized_cost_bps = slip_usd / traded_usd * 10_000 if traded_usd > 0 else 0.0

        # exit_z and exit_reason: trade_update has no Z context, so write
        # NULL exit_z and exit_reason='zero_cross' (the normal exit path).
        # EOM-driven exits get reason='eom' via the admin endpoint.
        from live.monitor.trade_journal import record_exit
        record_exit(conn, pair_id, exit_ts=exit_ts, exit_z=0.0,
                    exit_reason="zero_cross", realized_pnl=realized_pnl,
                    realized_cost_bps=realized_cost_bps)
        from live.state.persist import log_event
        log_event(conn, "position_closed", "INFO",
                  f"{pair_id} realized_pnl=${realized_pnl:+.2f} "
                  f"(A={pnl_a:+.2f}, B={pnl_b:+.2f})",
                  {"qty_a": qty_a, "qty_b": qty_b})
