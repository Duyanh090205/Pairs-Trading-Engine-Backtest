"""Drill: kill -9 mid-day. Restart must recover all state, no duplicate orders."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 3")
def test_open_positions_recovered():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_pending_orders_recovered():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_no_duplicate_orders_after_restart():
    raise NotImplementedError
