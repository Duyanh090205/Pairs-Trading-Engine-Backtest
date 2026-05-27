"""Pre-deployment hard gate. All 10 checks must pass before submit."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Allow `python live/preflight.py` from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class PreflightCheck:
    broker_auth_ok: bool = False
    websocket_connected: bool = False
    state_store_writable: bool = False
    state_store_recoverable: bool = False
    disconnect_drill_passed: bool = False
    reconcile_drill_passed: bool = False
    idempotency_test_passed: bool = False
    regime_features_buildable: bool = False
    dashboard_loads: bool = False
    cloud_deployment_ok: bool = False
    notes: list[str] = field(default_factory=list)

    def all_passed(self) -> bool:
        return all(v for k, v in asdict(self).items() if k != "notes")

    def summary(self) -> str:
        lines = ["Preflight check results:"]
        for k, v in asdict(self).items():
            if k == "notes":
                continue
            mark = "[+]" if v else "[-]"
            lines.append(f"  {mark} {k}")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def check_broker_auth(result: PreflightCheck) -> None:
    try:
        from live.broker.alpaca_client import AlpacaConfig, check_auth
        cfg = AlpacaConfig.from_env()
        info = check_auth(cfg)
        result.broker_auth_ok = info["status"] == "ACTIVE"
        result.notes.append(
            f"Account status={info['status']}, cash=${info['cash']:.2f}, equity=${info['equity']:.2f}"
        )
    except Exception as e:
        result.notes.append(f"broker_auth failed: {e}")


def check_state_store(result: PreflightCheck) -> None:
    try:
        from live.state.persist import connect, init_db, log_event
        db = Path(os.environ.get("STATE_DB_PATH", "./live/state/live_state.db"))
        init_db(db)
        with connect(db) as conn:
            log_event(conn, "preflight", "INFO", "write test")
            row = conn.execute(
                "SELECT message FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            result.state_store_writable = True
            result.state_store_recoverable = row["message"] == "write test"
    except Exception as e:
        result.notes.append(f"state_store failed: {e}")


def run_all() -> PreflightCheck:
    result = PreflightCheck()
    check_broker_auth(result)
    check_state_store(result)
    # websocket / drills / dashboard / cloud filled in later phases
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    res = run_all()
    print(res.summary())
    sys.exit(0 if res.all_passed() else 1)
