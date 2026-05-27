"""Discord webhook alert routing with tier + dedupe."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum

import requests


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class _DedupeEntry:
    ts: float


_recent: dict[str, _DedupeEntry] = {}
_DEDUPE_WINDOW_S = 300         # 5 minutes
_PRUNE_AFTER_S = 1_800         # drop keys older than 30 min (well past dedupe window)
_PRUNE_EVERY_N_CALLS = 200     # amortized O(1) prune
_call_counter = [0]            # mutable singleton


def _prune_recent(now: float) -> None:
    cutoff = now - _PRUNE_AFTER_S
    stale = [k for k, v in _recent.items() if v.ts < cutoff]
    for k in stale:
        del _recent[k]


def _recently_sent(key: str | None) -> bool:
    if not key:
        return False
    now = time.time()
    _call_counter[0] += 1
    if _call_counter[0] % _PRUNE_EVERY_N_CALLS == 0:
        _prune_recent(now)
    entry = _recent.get(key)
    if entry and (now - entry.ts) < _DEDUPE_WINDOW_S:
        return True
    _recent[key] = _DedupeEntry(ts=now)
    return False


def alert(severity: Severity, message: str, dedupe_key: str | None = None) -> bool:
    """Send Discord alert. Returns True if sent, False if suppressed."""
    if severity != Severity.INFO and _recently_sent(dedupe_key):
        return False
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    icon = {"CRITICAL": "🔴", "ERROR": "🟠", "WARN": "🟡", "INFO": "🟢"}[severity.value]
    payload = {"content": f"{icon} [{severity.value}] {message}"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code < 300
    except requests.RequestException:
        return False
