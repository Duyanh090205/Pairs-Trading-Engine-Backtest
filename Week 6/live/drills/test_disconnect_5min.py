"""GRADED drill: 5-min WebSocket cut. Engine must detect, halt entries, reconnect, reconcile, resume."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 3")
def test_disconnect_detected_within_30s():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_halt_new_entries_during_disconnect():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_reconnect_with_exponential_backoff():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_reconcile_positions_after_reconnect():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_resume_cleanly_no_duplicate_orders():
    raise NotImplementedError
