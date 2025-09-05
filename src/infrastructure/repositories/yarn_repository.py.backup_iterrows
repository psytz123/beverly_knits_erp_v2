"""Concrete implementation of yarn repository using existing data loaders."""

import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.domain.interfaces.yarn_repository import IYarnRepository
from src.domain.entities.yarn import Yarn
from src.data_loaders.unified_data_loader import UnifiedDataLoader
from src.utils.cache_manager import UnifiedCacheManager


class YarnRepository(IYarnRepository):
    """Repository implementation for yarn data access."""
    
    def __init__(self, data_loader: UnifiedDataLoader, cache_manager: UnifiedCacheManager):
        """Initialize repository with data loader and cache."""
        self.data_loader = data_loader
        self.cache = cache_manager
        self.logger = logging.getLogger(__name__)
        self._yarn_cache_key = "yarn_inventory_df"
        self._cache_ttl = 900  # 15 minutes
    
    async def _get_yarn_df(self) -> pd.DataFrame:
        """Get yarn DataFrame from cache or data loader."""
        # Try cache first
        cached_df = await self.cache.get(self._yarn_cache_key)
        if cached_df is not None:
            return cached_df
        
        # Load from data loader
        try:
            df = self.data_loader.load_yarn_inventory()
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Cache the DataFrame
            await self.cache.set(self._yarn_cache_key, df, ttl=self._cache_ttl)
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to load yarn inventory: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match entity fields."""
        column_mapping = {
            'Desc#': 'yarn_id',
            'desc_num': 'yarn_id',
            'YarnID': 'yarn_id',
            'Description': 'description',
            'Theoretical Balance': 'theoretical_balance',
            'theoretical_balance': 'theoretical_balance',
            'Allocated': 'allocated',
            'allocated': 'allocated',
            'On Order': 'on_order',
            'on_order': 'on_order',
            'Planning Balance': 'planning_balance',
            'Planning_Balance': 'planning_balance',
            'Min Stock': 'min_stock_level',
            'min_stock': 'min_stock_level',
            'Lead Time': 'lead_time_days',
            'lead_time': 'lead_time_days',
            'Color': 'color',
            'Type': 'yarn_type',
            'Supplier': 'supplier',
            'Cost': 'cost_per_unit'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['theoretical_balance', 'allocated', 'on_order', 
                          'min_stock_level', 'lead_time_days', 'cost_per_unit']
        
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and dollar signs if present
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate planning balance if not present
        if 'planning_balance' not in df.columns:
            if all(col in df.columns for col in ['theoretical_balance', 'allocated', 'on_order']):
                df['planning_balance'] = df['theoretical_balance'] + df['allocated'] + df['on_order']
        
        return df
    
    def _df_row_to_yarn(self, row: pd.Series) -> Yarn:
        """Convert DataFrame row to Yarn entity."""
        return Yarn(
            yarn_id=str(row.get('yarn_id', '')),
            description=str(row.get('description', '')),
            theoretical_balance=float(row.get('theoretical_balance', 0)),
            allocated=float(row.get('allocated', 0)),
            on_order=float(row.get('on_order', 0)),
            min_stock_level=float(row.get('min_stock_level', 0)),
            lead_time_days=int(row.get('lead_time_days', 14)),
            color=row.get('color'),
            yarn_type=row.get('yarn_type'),
            supplier=row.get('supplier'),
            cost_per_unit=float(row.get('cost_per_unit', 0)),
            last_updated=datetime.utcnow()
        )
    
    async def get_by_id(self, yarn_id: str) -> Optional[Yarn]:
        """Get a single yarn by ID."""
        df = await self._get_yarn_df()
        
        # Filter for the specific yarn
        yarn_df = df[df['yarn_id'] == yarn_id]
        
        if yarn_df.empty:
            return None
        
        return self._df_row_to_yarn(yarn_df.iloc[0])
    
    async def get_all(self, limit: int = 1000, offset: int = 0) -> List[Yarn]:
        """Get all yarns with pagination."""
        df = await self._get_yarn_df()
        
        # Apply pagination
        end = min(offset + limit, len(df))
        paginated_df = df.iloc[offset:end]
        
        return [self._df_row_to_yarn(row) for _, row in paginated_df.iterrows()]
    
    async def get_by_ids(self, yarn_ids: List[str]) -> List[Yarn]:
        """Get multiple yarns by their IDs."""
        df = await self._get_yarn_df()
        
        # Filter for the specific yarns
        yarn_df = df[df['yarn_id'].isin(yarn_ids)]
        
        return [self._df_row_to_yarn(row) for _, row in yarn_df.iterrows()]
    
    async def get_shortages(self, threshold: float = 0) -> List[Yarn]:
        """Get yarns with planning balance below threshold."""
        df = await self._get_yarn_df()
        
        # Filter for shortages
        shortage_df = df[df['planning_balance'] < threshold]
        
        # Sort by planning balance (most critical first)
        shortage_df = shortage_df.sort_values('planning_balance')
        
        return [self._df_row_to_yarn(row) for _, row in shortage_df.iterrows()]
    
    async def get_by_type(self, yarn_type: str) -> List[Yarn]:
        """Get yarns by type."""
        df = await self._get_yarn_df()
        
        if 'yarn_type' not in df.columns:
            return []
        
        type_df = df[df['yarn_type'] == yarn_type]
        
        return [self._df_row_to_yarn(row) for _, row in type_df.iterrows()]
    
    async def get_by_supplier(self, supplier: str) -> List[Yarn]:
        """Get yarns by supplier."""
        df = await self._get_yarn_df()
        
        if 'supplier' not in df.columns:
            return []
        
        supplier_df = df[df['supplier'] == supplier]
        
        return [self._df_row_to_yarn(row) for _, row in supplier_df.iterrows()]
    
    async def search(self, query: str) -> List[Yarn]:
        """Search yarns by description or ID."""
        df = await self._get_yarn_df()
        
        # Search in both yarn_id and description
        query_lower = query.lower()
        mask = (
            df['yarn_id'].astype(str).str.lower().str.contains(query_lower, na=False) |
            df['description'].astype(str).str.lower().str.contains(query_lower, na=False)
        )
        
        search_df = df[mask]
        
        return [self._df_row_to_yarn(row) for _, row in search_df.iterrows()]
    
    async def update(self, yarn: Yarn) -> bool:
        """Update yarn data."""
        # This would typically update the database
        # For now, we'll invalidate the cache to force reload
        await self.cache.delete(self._yarn_cache_key)
        
        # In a real implementation, this would update the data source
        self.logger.info(f"Yarn {yarn.yarn_id} marked for update (cache invalidated)")
        
        return True
    
    async def bulk_update(self, yarns: List[Yarn]) -> int:
        """Bulk update multiple yarns."""
        # Invalidate cache
        await self.cache.delete(self._yarn_cache_key)
        
        # In a real implementation, this would batch update the data source
        self.logger.info(f"Bulk update for {len(yarns)} yarns (cache invalidated)")
        
        return len(yarns)
    
    async def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all yarns."""
        df = await self._get_yarn_df()
        
        # Calculate summary statistics
        stats = {
            'total_yarns': len(df),
            'total_theoretical_balance': float(df['theoretical_balance'].sum()),
            'total_allocated': float(df['allocated'].sum()),
            'total_on_order': float(df['on_order'].sum()),
            'total_planning_balance': float(df['planning_balance'].sum()),
            'shortage_count': len(df[df['planning_balance'] < 0]),
            'critical_shortage_count': len(df[df['planning_balance'] < df['min_stock_level']]),
            'avg_planning_balance': float(df['planning_balance'].mean()),
            'min_planning_balance': float(df['planning_balance'].min()),
            'max_planning_balance': float(df['planning_balance'].max()),
            'yarns_by_type': df['yarn_type'].value_counts().to_dict() if 'yarn_type' in df.columns else {},
            'yarns_by_supplier': df['supplier'].value_counts().to_dict() if 'supplier' in df.columns else {},
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return stats
    
    async def get_yarns_needing_reorder(self, daily_usage_map: Dict[str, float]) -> List[Yarn]:
        """Get yarns that need to be reordered based on usage and lead time."""
        df = await self._get_yarn_df()
        
        yarns_to_reorder = []
        
        for yarn_id, daily_usage in daily_usage_map.items():
            yarn_row = df[df['yarn_id'] == yarn_id]
            
            if not yarn_row.empty:
                yarn = self._df_row_to_yarn(yarn_row.iloc[0])
                
                if yarn.needs_reorder(daily_usage):
                    yarns_to_reorder.append(yarn)
        
        # Sort by urgency (days of stock remaining)
        yarns_to_reorder.sort(
            key=lambda y: y.days_of_stock(daily_usage_map.get(y.yarn_id, 1)),
            reverse=False  # Least days first
        )
        
        return yarns_to_reorder
    
    async def refresh_from_source(self) -> bool:
        """Refresh data from source systems."""
        try:
            # Clear cache
            await self.cache.delete(self._yarn_cache_key)
            
            # Force reload
            await self._get_yarn_df()
            
            self.logger.info("Yarn inventory refreshed from source")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to refresh yarn inventory: {e}")
            return False