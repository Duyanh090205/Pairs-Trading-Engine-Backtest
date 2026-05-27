"""Phase 4 dashboard smoketest. Uses FastAPI TestClient — no network."""
from __future__ import annotations

import os
import sys
import tempfile
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


def _bootstrap_db_with_data():
    """Create a temp state DB with realistic test rows."""
    from datetime import datetime, timezone
    from live.state.persist import connect, init_db, log_event
    td = tempfile.mkdtemp(prefix="dash_smoke_")
    db = Path(td) / "state.db"
    init_db(db)
    ts = datetime.now(timezone.utc).isoformat()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
            "notional_a, notional_b, entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("JPM_BAC", "JPM", "BAC", 1.2, 1, 7500, 9000, ts, 3.15),
        )
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, submitted_ts, decision_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("coid_x", "JPM_BAC", ts, "JPM", "buy", 100, "market", "filled", ts, 190.0),
        )
        conn.execute(
            "INSERT INTO regime_decisions (month, stress_z, q67_thresh, halted, decided_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-06", 0.5, 0.7, 0, ts),
        )
        log_event(conn, "smoketest", "INFO", "dashboard bootstrap")
    return td, db


def _make_client(db: Path):
    from fastapi.testclient import TestClient
    os.environ["STATE_DB_PATH"] = str(db)
    # Force re-import so server picks up env var
    import importlib
    import live.dashboard.server
    importlib.reload(live.dashboard.server)
    return TestClient(live.dashboard.server.app)


def t_health():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["db_exists"] is True


def t_status():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/status")
    assert r.status_code == 200
    b = r.json()
    assert "kill_switch_halted" in b and "hardstop_tripped" in b


def t_positions_with_data():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/positions")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "JPM_BAC"


def t_orders():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/orders")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "JPM" and rows[0]["status"] == "filled"


def t_pnl_returns_zero_no_closed():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/pnl")
    assert r.status_code == 200
    b = r.json()
    assert b["realized_pnl_usd"] == 0.0
    assert b["open_positions"] == 1
    assert b["backtest_sharpe"] == 1.28


def t_regime_visible():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/regime")
    rows = r.json()["rows"]
    assert len(rows) == 1 and rows[0]["month"] == "2026-06"


def t_log_visible():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/api/log")
    rows = r.json()["rows"]
    assert any(row["event"] == "smoketest" for row in rows)


def t_index_renders_html():
    _, db = _bootstrap_db_with_data()
    c = _make_client(db)
    r = c.get("/")
    assert r.status_code == 200
    assert "Week 6 Live Paper Trading" in r.text
    assert "htmx" in r.text


def main() -> int:
    print(f"{YELLOW}== Phase 4 Dashboard Smoketest =={RESET}")
    for name, fn in [
        ("health", t_health),
        ("status", t_status),
        ("positions_with_data", t_positions_with_data),
        ("orders", t_orders),
        ("pnl", t_pnl_returns_zero_no_closed),
        ("regime", t_regime_visible),
        ("log", t_log_visible),
        ("index_html", t_index_renders_html),
    ]:
        check(name, fn)
    if failures:
        print(f"{RED}{len(failures)} FAILED:{RESET} {failures}")
        return 1
    print(f"{GREEN}All Phase 4 smoketests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
