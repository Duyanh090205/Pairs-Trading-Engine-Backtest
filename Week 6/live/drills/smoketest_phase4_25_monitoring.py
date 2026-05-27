"""Phase 4.25 monitoring smoketest — kill_switch evaluate, trade_journal, drift, anomaly, reports, live_vs_backtest."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
failures: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"{GREEN}PASS{RESET}  {name}")
    except Exception as e:
        print(f"{RED}FAIL{RESET}  {name}: {type(e).__name__}: {e}")
        failures.append(name)


def _seed_db():
    """DB with: 5 closed positions (3 losers, 2 winners) + 1 open."""
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="monitor_smoke_")
    db = Path(td) / "state.db"
    init_db(db)
    ts_open = datetime(2026, 5, 25, 13, 0, tzinfo=timezone.utc).isoformat()
    closed_pnls = [-50.0, -30.0, +120.0, -80.0, +200.0]   # 3 losses, 2 wins
    with connect(db) as conn:
        for i, p in enumerate(closed_pnls):
            entry_ts = datetime(2026, 5, 20 + i, 13, tzinfo=timezone.utc).isoformat()
            exit_ts = datetime(2026, 5, 20 + i, 16, tzinfo=timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
                "notional_a, notional_b, entry_ts, entry_z, exit_ts, exit_z, exit_reason, "
                "realized_pnl, predicted_cost_bps, realized_cost_bps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"P{i}_Q{i}", f"P{i}", f"Q{i}", 1.0, 1, 5000, 5000,
                 entry_ts, 3.1, exit_ts, 0.1, "zero_cross", p, 22.0, 22.0 + i * 2),
            )
        conn.execute(
            "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
            "notional_a, notional_b, entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("OPEN_X", "X", "Y", 1.0, 1, 5000, 5000, ts_open, 3.05),
        )
    return db


def t_kill_switch_no_trigger():
    from live.monitor.kill_switch import KillSwitchThresholds, evaluate
    from live.state.persist import connect
    db = _seed_db()
    with connect(db) as conn:
        # equity dropped 1%, well below 3% threshold
        r = evaluate(conn, session_start_equity=100_000.0, current_equity=99_000.0,
                     thresholds=KillSwitchThresholds())
    assert r is None, f"expected no trip, got {r}"


def t_kill_switch_drawdown_trip():
    from live.monitor.kill_switch import evaluate
    from live.state.persist import connect
    db = _seed_db()
    with connect(db) as conn:
        r = evaluate(conn, session_start_equity=100_000.0, current_equity=96_000.0)
    assert r is not None and "daily_drawdown" in r, f"got {r}"


def t_kill_switch_consec_losses():
    """All 10 most recent positions are losses → trip."""
    from live.monitor.kill_switch import KillSwitchThresholds, evaluate
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="ks_consec_")
    db = Path(td) / "state.db"
    init_db(db)
    with connect(db) as conn:
        for i in range(12):
            ts = datetime(2026, 5, 1 + i, 13, tzinfo=timezone.utc).isoformat()
            ex = datetime(2026, 5, 1 + i, 16, tzinfo=timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
                "notional_a, notional_b, entry_ts, entry_z, exit_ts, exit_z, exit_reason, realized_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"L{i}_M{i}", f"L{i}", f"M{i}", 1.0, 1, 5000, 5000,
                 ts, 3.1, ex, 0.1, "zero_cross", -10.0),
            )
        r = evaluate(conn, session_start_equity=100_000.0, current_equity=100_000.0,
                     thresholds=KillSwitchThresholds(consecutive_losses_max=10))
    assert r is not None and "consecutive_losses" in r, r


def t_trade_journal_records_entry_exit():
    from live.monitor.trade_journal import list_recent, record_entry, record_exit
    from live.state.persist import connect
    db = _seed_db()
    with connect(db) as conn:
        record_entry(conn, "OPEN_X", predicted_cost_bps=20.0)
        record_exit(conn, "OPEN_X", exit_ts="2026-05-25T16:00:00+00:00",
                    exit_z=0.05, exit_reason="zero_cross",
                    realized_pnl=15.0, realized_cost_bps=23.5)
        attrs = list_recent(conn, limit=10)
    open_x = next((a for a in attrs if a.pair_id == "OPEN_X"), None)
    assert open_x is not None
    assert open_x.predicted_cost_bps == 20.0
    assert open_x.realized_cost_bps == 23.5
    assert abs(open_x.drift_bps - 3.5) < 1e-9


def t_drift_monitor_returns_mean():
    from live.monitor.drift_monitor import compute_drift_bps
    from live.state.persist import connect, init_db
    # Need recent (last 21 days) trades — _seed_db has 2026-05-2x trades; only counts
    # those within now-21d window. Build a fresh DB with today-ish dates.
    from datetime import timedelta
    td = tempfile.mkdtemp(prefix="drift_test_")
    db = Path(td) / "state.db"
    init_db(db)
    with connect(db) as conn:
        for i in range(5):
            t = (datetime.now(timezone.utc) - timedelta(days=i + 1))
            t_exit = t + timedelta(hours=3)
            conn.execute(
                "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
                "notional_a, notional_b, entry_ts, entry_z, exit_ts, exit_z, exit_reason, "
                "realized_pnl, predicted_cost_bps, realized_cost_bps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"P{i}_Q{i}", f"P{i}", f"Q{i}", 1.0, 1, 5000, 5000,
                 t.isoformat(), 3.1, t_exit.isoformat(), 0.1, "zero_cross",
                 10.0, 20.0, 25.0),  # realized 5bps worse than predicted
            )
        d = compute_drift_bps(conn, lookback_days=21)
    assert d["n_trades"] == 5
    assert abs(d["mean_drift_bps"] - 5.0) < 1e-9
    assert d["total_cost_drift_usd"] > 0


def t_anomaly_detector_flags_slippage():
    from live.monitor.anomaly_detector import AnomalyThresholds, check_fill
    flags = check_fill(100.0, 100.6, "buy", AnomalyThresholds(max_slippage_bps=50))
    assert any("slippage" in f for f in flags), flags
    clean = check_fill(100.0, 100.05, "buy")
    assert clean == [], clean
    # SELL adverse: filled lower than decision
    flags2 = check_fill(100.0, 99.4, "sell", AnomalyThresholds(max_slippage_bps=50))
    assert any("slippage" in f for f in flags2), flags2


def t_live_vs_backtest_metrics():
    from live.monitor.live_vs_backtest import rolling_comparison
    from live.state.persist import connect
    db = _seed_db()
    with connect(db) as conn:
        rows = rolling_comparison(conn, window_days=365)   # wide window to include seed data
    # At least the reference rows should be present
    metrics = {r.metric for r in rows}
    assert "win_rate" in metrics
    assert "sharpe_per_fold" in metrics
    n_trades_row = next((r for r in rows if r.metric == "n_trades"), None)
    assert n_trades_row is not None


def t_reports_generate():
    from live.monitor.reports import daily_summary, month_end_summary, weekly_summary
    from live.state.persist import connect
    db = _seed_db()
    with connect(db) as conn:
        daily = daily_summary(conn, date(2026, 5, 22))
        weekly = weekly_summary(conn, date(2026, 5, 24))
        monthly = month_end_summary(conn, "2026-05")
    assert "Daily Summary" in daily
    assert "Weekly Summary" in weekly
    assert "Month-End Summary" in monthly


def main() -> int:
    print(f"{YELLOW}== Phase 4.25 Monitoring Smoketest =={RESET}")
    for n, f in [
        ("kill_switch_no_trigger", t_kill_switch_no_trigger),
        ("kill_switch_drawdown_trip", t_kill_switch_drawdown_trip),
        ("kill_switch_consec_losses", t_kill_switch_consec_losses),
        ("trade_journal_record", t_trade_journal_records_entry_exit),
        ("drift_monitor_mean", t_drift_monitor_returns_mean),
        ("anomaly_detector_slippage", t_anomaly_detector_flags_slippage),
        ("live_vs_backtest_metrics", t_live_vs_backtest_metrics),
        ("reports_generate", t_reports_generate),
    ]:
        check(n, f)
    if failures:
        print(f"{RED}{len(failures)} FAILED:{RESET} {failures}")
        return 1
    print(f"{GREEN}All Phase 4.25 smoketests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
