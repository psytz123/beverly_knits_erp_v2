"""
Contract test placeholders for inventory service parity.

These tests will be completed once the microservice exposes real endpoints.
"""

import os
import pytest

MONOLITH_URL = os.getenv("MONOLITH_BASE_URL", "http://localhost:5006")
SERVICE_URL = os.getenv("SERVICE_BASE_URL", "http://localhost:8001")


@pytest.mark.skip(reason="Pending implementation of inventory service endpoints")
def test_inventory_contract_parity():
    """Compare monolith vs microservice responses for inventory list endpoint."""
    assert MONOLITH_URL
    assert SERVICE_URL
