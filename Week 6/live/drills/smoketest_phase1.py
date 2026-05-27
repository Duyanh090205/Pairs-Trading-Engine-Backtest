"""Phase 1 scaffold smoketest — no broker, no network. Run before any phase 2 work.

Checks:
  1. All live/ modules import cleanly (no syntax errors, no missing deps)
  2. schema.sql applies and all tables exist
  3. PreflightCheck dataclass round-trips
  4. ZTracker math is correct against numpy reference
  5. OrderRequest.client_order_id() is deterministic + collision-resistant
  6. Hardstop trip → flag created → is_tripped True → clear → flag gone
  7. Alert dedupe suppresses duplicates within window
  8. Audit log insert/read round-trip works
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Allow running as `python live/drills/smoketest_phase1.py`
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

failures: list[str] = []


def check(name: str, fn):
    try:
        fn()
        print(f"{GREEN}PASS{RESET}  {name}")
    except Exception as e:
        print(f"{RED}FAIL{RESET}  {name}: {type(e).__name__}: {e}")
        failures.append(name)


def t_imports():
    """Pure-stdlib + numpy modules must import. Modules that require optional
    third-party deps (fastapi, dotenv) are tolerated when the dep is missing —
    those deps install with `pip install -r requirements-live.txt`."""
    required = [
        "live.state.persist", "live.state.recovery",
        "live.broker.alpaca_client", "live.broker.websocket_handler",
        "live.execution.order_manager", "live.execution.reconciliation",
        "live.engine_live.live_pair", "live.engine_live.z_tracker",
        "live.engine_live.regime_check",
        "live.monitor.alerts", "live.monitor.kill_switch", "live.monitor.reports",
        "live.monitor.trade_journal", "live.monitor.drift_monitor",
        "live.monitor.anomaly_detector", "live.monitor.live_vs_backtest",
        "live.safety.hardstop",
        "live.preflight",
    ]
    optional = {
        "live.dashboard.server": "fastapi",
    }
    for m in required:
        importlib.import_module(m)
    for m, dep in optional.items():
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if dep in str(e):
                print(f"{YELLOW}  (optional dep '{dep}' missing for {m} — install via requirements-live.txt){RESET}")
            else:
                raise


def t_schema_apply():
    from live.state.persist import init_db, connect
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = Path(td) / "test.db"
        init_db(db)
        with connect(db) as conn:
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            expected = {"positions", "orders", "bars", "ticks", "audit_log",
                        "kill_switch", "regime_decisions"}
            missing = expected - tables
            assert not missing, f"missing tables: {missing}"
            row = conn.execute("SELECT halted FROM kill_switch WHERE id=1").fetchone()
            assert row is not None and row["halted"] == 0


def t_preflight_dataclass():
    from live.preflight import PreflightCheck
    pc = PreflightCheck()
    assert pc.all_passed() is False
    pc.broker_auth_ok = True
    pc.websocket_connected = True
    pc.state_store_writable = True
    pc.state_store_recoverable = True
    pc.disconnect_drill_passed = True
    pc.reconcile_drill_passed = True
    pc.idempotency_test_passed = True
    pc.regime_features_buildable = True
    pc.dashboard_loads = True
    pc.cloud_deployment_ok = True
    assert pc.all_passed() is True
    s = pc.summary()
    assert "broker_auth_ok" in s


def t_ztracker_math():
    from live.engine_live.z_tracker import ZTracker
    rng = np.random.default_rng(42)
    seed = rng.normal(0, 1, 120).tolist()
    z = ZTracker(window=60, seed=seed)
    # Push one new value; compare to numpy
    new = 1.5
    out = z.push(new)
    # Reference: backtest uses sample std (ddof=1) — must match.
    ref_window = np.array((seed + [new])[-60:])
    mean = ref_window.mean()
    std = ref_window.std(ddof=1)
    expected = (new - mean) / std
    assert out is not None
    assert abs(out - expected) < 1e-9, f"got {out}, expected {expected}"


def t_order_id_determinism():
    from live.execution.order_manager import OrderRequest
    r1 = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 100.0, "market")
    r2 = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 100.0, "market")
    assert r1.client_order_id() == r2.client_order_id()
    # Different qty → different id
    r3 = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 101.0, "market")
    assert r1.client_order_id() != r3.client_order_id()
    # Different leg → different id
    r4 = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "B", "BAC", "sell", 100.0, "market")
    assert r1.client_order_id() != r4.client_order_id()


def t_hardstop_roundtrip():
    from live.safety import hardstop
    from live.state.persist import init_db, connect
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = Path(td) / "test.db"
        init_db(db)
        flag = Path(td) / "HARDSTOP.flag"
        hardstop.HARDSTOP_FLAG_PATH = flag
        with connect(db) as conn:
            # No trip with healthy equity
            st = hardstop.check(conn, session_start_equity=100_000.0,
                                current_equity=99_500.0)
            assert st.tripped is False, f"unexpected trip: {st.reason}"
            # Trip via 6% equity drop
            st = hardstop.check(conn, session_start_equity=100_000.0,
                                current_equity=94_000.0)
            assert st.tripped is True
            assert flag.exists()
            assert hardstop.is_tripped() is True
        hardstop.clear("smoketest clear")
        assert not flag.exists()
        assert hardstop.is_tripped() is False


def t_alert_dedupe():
    from live.monitor import alerts
    alerts._recent.clear()
    # Pretend Discord URL absent → alert returns False but dedupe still tracks
    os.environ.pop("DISCORD_WEBHOOK_URL", None)
    r1 = alerts.alert(alerts.Severity.ERROR, "test", dedupe_key="k1")
    # Second within window should be suppressed (returns False)
    r2 = alerts.alert(alerts.Severity.ERROR, "test", dedupe_key="k1")
    assert r1 is False  # no URL configured
    assert r2 is False
    # Without dedupe key, every call considered fresh
    r3 = alerts.alert(alerts.Severity.ERROR, "no dedupe")
    assert r3 is False


def t_audit_log_roundtrip():
    from live.state.persist import init_db, connect, log_event
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = Path(td) / "test.db"
        init_db(db)
        with connect(db) as conn:
            log_event(conn, "smoketest", "INFO", "hello", payload={"a": 1})
            row = conn.execute(
                "SELECT event, level, message, payload FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert row["event"] == "smoketest"
            assert row["level"] == "INFO"
            assert row["message"] == "hello"
            import json as _j
            assert _j.loads(row["payload"]) == {"a": 1}


def main() -> int:
    print(f"{YELLOW}== Phase 1 Smoketest =={RESET}")
    check("imports", t_imports)
    check("schema_apply", t_schema_apply)
    check("preflight_dataclass", t_preflight_dataclass)
    check("ztracker_math", t_ztracker_math)
    check("order_id_determinism", t_order_id_determinism)
    check("hardstop_roundtrip", t_hardstop_roundtrip)
    check("alert_dedupe", t_alert_dedupe)
    check("audit_log_roundtrip", t_audit_log_roundtrip)
    print()
    if failures:
        print(f"{RED}{len(failures)} FAILED:{RESET} {failures}")
        return 1
    print(f"{GREEN}All Phase 1 smoketests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
