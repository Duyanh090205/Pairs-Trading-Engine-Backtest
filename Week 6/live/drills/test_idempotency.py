"""Drill: re-submit same client_order_id. Broker must dedup, no double order."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 3")
def test_duplicate_client_order_id_returns_existing():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3")
def test_client_order_id_deterministic():
    raise NotImplementedError
