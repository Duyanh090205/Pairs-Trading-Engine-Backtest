"""Single source of truth for computing a position's realized P&L from fills.

Why this module exists
----------------------
Realized P&L was historically computed inline in three places (the live
``trading_stream`` close handler, the ``finalize_eom_positions`` admin path,
and ad-hoc). Each summed fills for a leg by ``ticker + side`` with a loose (or
absent) timestamp filter. That is correct only for a pair's FIRST lifecycle.

For any pair traded more than once (e.g. CRWD_EQT in May and again in July),
the entry-side sum swept in the PRIOR lifecycle's same-side fills — a CRWD
"buy" that was really a short-cover from the earlier round got averaged into
the later round's entry price, corrupting it and inflating the loss. Observed
live: CRWD_EQT July stored -$532.21 vs true -$107.35 (a $425 phantom loss).

The fix is to scope every leg's fills to the *lifecycle window*:

    entry fills:  prior_exit_ts <  filled_ts <= entry_ts   (entry side)
    exit  fills:  entry_ts      <  filled_ts <  next_entry_ts  (exit side)

where ``prior_exit_ts`` = the most recent exit_ts of a CLOSED position for the
same pair before this entry, and ``next_entry_ts`` = the earliest entry_ts of a
LATER position for the same pair. Either bound is ``None`` for the first / last
lifecycle, in which case that side is unbounded (the original behaviour, which
is correct when no other lifecycle exists — e.g. the live path closing the only
open position).
"""
from __future__ import annotations

import sqlite3
from typing import Optional


def lifecycle_realized_pnl(
    conn: sqlite3.Connection,
    pair_id: str,
    side_a: str,
    side_b: str,
    direction: int,
    entry_ts: str,
    exit_ts: Optional[str] = None,
) -> Optional[dict]:
    """Realized P&L for ONE position lifecycle, from actual broker fills.

    Scoped to the current lifecycle so a prior/next lifecycle of the SAME pair
    cannot contaminate the averaged fill prices (see module docstring).

    ``exit_ts`` is the position's recorded close time = ``max`` of its exit-fill
    timestamps. Pass it when recomputing a CLOSED position so exit fills are
    upper-bounded to this lifecycle (a later lifecycle's entry fills can land a
    few seconds before its own recorded entry_ts and would otherwise leak in).
    Leave it ``None`` on the live close path — the position being closed is the
    latest lifecycle, so there are no later fills to exclude.

    Returns a dict with ``realized_pnl``, ``realized_cost_bps``, ``exit_ts`` and
    per-leg breakdown, or ``None`` if either exit leg has no fills yet (position
    not fully closed) — callers treat ``None`` as "not ready, leave open".
    """
    entry_side_a = "buy" if direction == 1 else "sell"
    entry_side_b = "sell" if direction == 1 else "buy"
    exit_side_a = "sell" if direction == 1 else "buy"
    exit_side_b = "buy" if direction == 1 else "sell"

    # Lower bound for entry fills: the prior lifecycle's close for this pair.
    prow = conn.execute(
        "SELECT MAX(exit_ts) AS x FROM positions "
        "WHERE pair_id = ? AND exit_ts IS NOT NULL AND exit_ts < ?",
        (pair_id, entry_ts),
    ).fetchone()
    prior_exit_ts = prow["x"] if prow and prow["x"] else None

    # Legacy fills (pre-timestamp-normalization) have filled_ts=NULL and so are
    # invisible to any timestamp bound. They only exist on the pair's FIRST
    # lifecycle (all fills recorded since then carry a timestamp), so attribute
    # them to the first lifecycle — the one with no prior close.
    include_null = prior_exit_ts is None

    def _sum(ticker: str, side: str, ts_op: str, ts_ref: str,
             ts_lower: Optional[str], ts_upper: Optional[str]):
        """Qty, qty-weighted fill price, qty-weighted decision price, max ts."""
        cond = "filled_ts " + ts_op + " ?"
        params = [pair_id, ticker, side, ts_ref]
        if ts_lower is not None:
            cond += " AND filled_ts > ?"
            params.append(ts_lower)
        if ts_upper is not None:
            # Inclusive: ts_upper is a lifecycle's exit_ts = MAX of its own exit
            # fill timestamps, so the latest exit fill must be kept.
            cond += " AND filled_ts <= ?"
            params.append(ts_upper)
        if include_null:
            cond = "((" + cond + ") OR filled_ts IS NULL)"
        sql = (
            "SELECT SUM(fill_qty) AS q, "
            "SUM(fill_qty * fill_price) / NULLIF(SUM(fill_qty), 0) AS px, "
            "SUM(fill_qty * decision_price) / NULLIF(SUM(fill_qty), 0) AS dpx, "
            "MAX(filled_ts) AS ts "
            "FROM orders WHERE pair_id = ? AND ticker = ? AND side = ? "
            "AND status = 'filled' AND " + cond
        )
        r = conn.execute(sql, params).fetchone()
        return (
            float(r["q"]) if r and r["q"] else 0.0,
            float(r["px"]) if r and r["px"] else 0.0,
            float(r["dpx"]) if r and r["dpx"] else 0.0,
            r["ts"] if r else None,
        )

    # Entry fills at/before entry_ts, after the prior lifecycle's close.
    eqa, epa, edpa, _ = _sum(side_a, entry_side_a, "<=", entry_ts, prior_exit_ts, None)
    eqb, epb, edpb, _ = _sum(side_b, entry_side_b, "<=", entry_ts, prior_exit_ts, None)
    # Exit fills strictly after entry_ts, up to this lifecycle's close (exit_ts).
    xqa, xpa, xdpa, xtsa = _sum(side_a, exit_side_a, ">", entry_ts, None, exit_ts)
    xqb, xpb, xdpb, xtsb = _sum(side_b, exit_side_b, ">", entry_ts, None, exit_ts)

    # Both exit legs (and both entry legs, as a sanity check) must be present.
    if xqa <= 0 or xqb <= 0 or eqa <= 0 or eqb <= 0:
        return None

    qty_a = min(eqa, xqa)
    qty_b = min(eqb, xqb)
    if direction == 1:
        pnl_a = qty_a * (xpa - epa)         # long A
        pnl_b = qty_b * (epb - xpb)         # short B
    else:
        pnl_a = qty_a * (epa - xpa)         # short A
        pnl_b = qty_b * (xpb - epb)         # long B
    realized_pnl = pnl_a + pnl_b

    exit_ts = max([t for t in (xtsa, xtsb) if t], default=None)

    # Realized execution cost = broker slippage vs decision price across the 4
    # legs, normalized to traded notional (bps). Convention: BUY filled above
    # decision = positive cost; SELL filled below decision = positive cost.
    # Legs with decision_price NULL/0 (legacy rows) are excluded from both
    # numerator and denominator.
    slip_usd = 0.0
    traded_usd = 0.0
    for side, q, fp, dp in (
        (entry_side_a, qty_a, epa, edpa), (entry_side_b, qty_b, epb, edpb),
        (exit_side_a, qty_a, xpa, xdpa), (exit_side_b, qty_b, xpb, xdpb),
    ):
        if q <= 0 or fp <= 0 or dp <= 0:
            continue
        slip_usd += q * ((fp - dp) if side == "buy" else (dp - fp))
        traded_usd += q * dp
    realized_cost_bps = slip_usd / traded_usd * 10_000 if traded_usd > 0 else 0.0

    return {
        "realized_pnl": realized_pnl,
        "realized_cost_bps": realized_cost_bps,
        "exit_ts": exit_ts,
        "qty_a": qty_a, "qty_b": qty_b,
        "pnl_a": pnl_a, "pnl_b": pnl_b,
        "entry_px_a": epa, "exit_px_a": xpa,
        "entry_px_b": epb, "exit_px_b": xpb,
    }
