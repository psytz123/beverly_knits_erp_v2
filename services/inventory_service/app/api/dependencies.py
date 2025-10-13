"""Dependency providers for the inventory service API layer."""

from __future__ import annotations

from functools import lru_cache

from ..domain.service import InventoryDomainService


@lru_cache(maxsize=1)
def _service_singleton() -> InventoryDomainService:
    """Lazy-initialized domain service singleton."""
    return InventoryDomainService()


def get_inventory_service() -> InventoryDomainService:
    """Return the domain service instance."""
    return _service_singleton()
