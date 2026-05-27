"""Auto-generated daily / weekly / month-end reports."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone


def daily_summary(conn: sqlite3.Connection, day: date) -> str:
    day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    day_end = datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).isoformat()
    closed = conn.execute(
        "SELECT pair_id, realized_pnl, realized_cost_bps, exit_reason "
        "FROM positions WHERE exit_ts BETWEEN ? AND ?",
        (day_start, day_end),
    ).fetchall()
    opened = conn.execute(
        "SELECT pair_id, entry_z FROM positions WHERE entry_ts BETWEEN ? AND ?",
        (day_start, day_end),
    ).fetchall()
    open_now = conn.execute(
        "SELECT COUNT(*) AS n FROM positions WHERE exit_ts IS NULL"
    ).fetchone()
    alerts = conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE level IN ('CRITICAL','ERROR') "
        "AND ts BETWEEN ? AND ?", (day_start, day_end),
    ).fetchone()
    total_pnl = sum(float(r["realized_pnl"] or 0) for r in closed)
    lines = [
        f"# Daily Summary — {day.isoformat()}",
        "",
        f"- Trades opened: {len(opened)}",
        f"- Trades closed: {len(closed)}",
        f"- Realized P&L: ${total_pnl:.2f}",
        f"- Currently open: {open_now['n']} pair(s)",
        f"- Alerts (CRITICAL/ERROR): {alerts['n']}",
    ]
    if closed:
        lines.append("\n## Closed today")
        for r in closed:
            cost = r["realized_cost_bps"]
            lines.append(f"- {r['pair_id']}: PnL=${float(r['realized_pnl'] or 0):.2f} "
                         f"cost={cost:.1f}bps reason={r['exit_reason']}"
                         if cost is not None else
                         f"- {r['pair_id']}: PnL=${float(r['realized_pnl'] or 0):.2f} reason={r['exit_reason']}")
    return "\n".join(lines)


def weekly_summary(conn: sqlite3.Connection, week_ending: date) -> str:
    week_start = week_ending - timedelta(days=6)
    ws = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    we = datetime.combine(week_ending, datetime.max.time(), tzinfo=timezone.utc).isoformat()
    closed = conn.execute(
        "SELECT realized_pnl FROM positions WHERE exit_ts BETWEEN ? AND ?",
        (ws, we),
    ).fetchall()
    pnls = [float(r["realized_pnl"] or 0) for r in closed]
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return "\n".join([
        f"# Weekly Summary — week ending {week_ending.isoformat()}",
        "",
        f"- Trades closed: {len(pnls)}",
        f"- Wins: {wins}",
        f"- Realized P&L: ${total:.2f}",
        f"- Win rate: {wins / len(pnls):.1%}" if pnls else "- Win rate: n/a",
    ])


def month_end_summary(conn: sqlite3.Connection, month: str) -> str:
    """`month` in 'YYYY-MM' form."""
    closed = conn.execute(
        "SELECT realized_pnl, exit_reason FROM positions "
        "WHERE substr(exit_ts, 1, 7) = ?", (month,),
    ).fetchall()
    pnls = [float(r["realized_pnl"] or 0) for r in closed]
    regime = conn.execute(
        "SELECT halted FROM regime_decisions WHERE month = ?", (month,),
    ).fetchone()
    halted = bool(regime["halted"]) if regime else False
    return "\n".join([
        f"# Month-End Summary — {month}",
        "",
        f"- Regime halted: {halted}",
        f"- Trades closed: {len(pnls)}",
        f"- Realized P&L: ${sum(pnls):.2f}",
    ])
