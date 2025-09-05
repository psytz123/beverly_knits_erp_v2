"""Repository interface for yarn data access."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..entities.yarn import Yarn


class IYarnRepository(ABC):
    """Abstract interface for yarn repository operations."""
    
    @abstractmethod
    async def get_by_id(self, yarn_id: str) -> Optional[Yarn]:
        """Get a single yarn by ID."""
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 1000, offset: int = 0) -> List[Yarn]:
        """Get all yarns with pagination."""
        pass
    
    @abstractmethod
    async def get_by_ids(self, yarn_ids: List[str]) -> List[Yarn]:
        """Get multiple yarns by their IDs."""
        pass
    
    @abstractmethod
    async def get_shortages(self, threshold: float = 0) -> List[Yarn]:
        """Get yarns with planning balance below threshold."""
        pass
    
    @abstractmethod
    async def get_by_type(self, yarn_type: str) -> List[Yarn]:
        """Get yarns by type."""
        pass
    
    @abstractmethod
    async def get_by_supplier(self, supplier: str) -> List[Yarn]:
        """Get yarns by supplier."""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Yarn]:
        """Search yarns by description or ID."""
        pass
    
    @abstractmethod
    async def update(self, yarn: Yarn) -> bool:
        """Update yarn data."""
        pass
    
    @abstractmethod
    async def bulk_update(self, yarns: List[Yarn]) -> int:
        """Bulk update multiple yarns. Returns count of updated records."""
        pass
    
    @abstractmethod
    async def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all yarns."""
        pass
    
    @abstractmethod
    async def get_yarns_needing_reorder(self, daily_usage_map: Dict[str, float]) -> List[Yarn]:
        """Get yarns that need to be reordered based on usage and lead time."""
        pass
    
    @abstractmethod
    async def refresh_from_source(self) -> bool:
        """Refresh data from source systems."""
        pass