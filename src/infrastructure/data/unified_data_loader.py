"""Unified Data Loader - Single source of truth for all data loading."""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.data.column_mapper import ColumnMapper
from src.infrastructure.data.data_validator import DataValidator
from src.infrastructure.cache.multi_tier_cache import MultiTierCache
from src.infrastructure.data.exceptions import DataSourceException, DataLoadException, DataValidationException


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def load(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load data of specified type."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available."""
        pass


class FileDataSource(DataSource):
    """File-based data source."""
    
    def __init__(self, base_path: str):
        """Initialize file data source."""
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
        # File mappings for different data types
        self.file_mappings = {
            'yarn_inventory': 'yarn_inventory.xlsx',
            'bom_data': 'BOM_updated.csv',
            'production_orders': 'eFab_Knit_Orders.csv',
            'work_centers': 'QuadS_greigeFabricList_(1).xlsx',
            'machine_report': 'Machine Report fin1.csv',
            'sales_activity': 'Sales Activity Report.csv',
            'demand_data': '8-28-2025/Yarn_Demand_8_28_2025.xlsx'
        }
    
    def is_available(self) -> bool:
        """Check if base path exists."""
        return self.base_path.exists()
    
    async def load(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from file system."""
        if data_type not in self.file_mappings:
            raise DataLoadException(f"Unknown data type: {data_type}")
        
        file_name = self.file_mappings[data_type]
        
        # Check multiple possible locations
        possible_paths = [
            self.base_path / file_name,
            self.base_path / '8-28-2025' / file_name,
            self.base_path / 'current' / file_name
        ]
        
        for file_path in possible_paths:
            if file_path.exists():
                try:
                    self.logger.info(f"Loading {data_type} from {file_path}")
                    
                    # Load based on file extension
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path, low_memory=False)
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        raise DataLoadException(f"Unsupported file type: {file_path.suffix}")
                    
                    self.logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                    return df
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        raise DataSourceException(f"File not found for {data_type}")


class APIDataSource(DataSource):
    """API-based data source."""
    
    def __init__(self, base_url: str, api_key: str = None):
        """Initialize API data source."""
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    def is_available(self) -> bool:
        """Check if API is reachable."""
        if not self.base_url:
            return False
        
        # Could implement a health check here
        return True
    
    async def load(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from API."""
        import aiohttp
        
        endpoint_map = {
            'yarn_inventory': '/api/yarn/inventory',
            'production_orders': '/api/production/orders',
            'work_centers': '/api/production/work-centers'
        }
        
        if data_type not in endpoint_map:
            return None
        
        endpoint = endpoint_map[data_type]
        url = f"{self.base_url}{endpoint}"
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data)
                        self.logger.info(f"Loaded {len(df)} rows from API for {data_type}")
                        return df
                    else:
                        self.logger.error(f"API returned status {response.status} for {data_type}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"API load failed for {data_type}: {e}")
            return None


class DatabaseDataSource(DataSource):
    """Database data source."""
    
    def __init__(self, connection_string: str):
        """Initialize database data source."""
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        self.engine = None
    
    def is_available(self) -> bool:
        """Check if database is reachable."""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def load(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from database."""
        from sqlalchemy import create_engine
        
        query_map = {
            'yarn_inventory': "SELECT * FROM yarn_inventory WHERE status = 'active'",
            'bom_data': "SELECT * FROM bill_of_materials",
            'production_orders': "SELECT * FROM production_orders WHERE status != 'cancelled'"
        }
        
        if data_type not in query_map:
            return None
        
        query = query_map[data_type]
        
        try:
            if not self.engine:
                self.engine = create_engine(self.connection_string)
            
            df = pd.read_sql_query(query, self.engine)
            self.logger.info(f"Loaded {len(df)} rows from database for {data_type}")
            return df
            
        except Exception as e:
            self.logger.error(f"Database load failed for {data_type}: {e}")
            return None


class UnifiedDataLoader:
    """
    Single source of truth for all data loading.
    Implements fallback strategy and caching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified data loader."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cache = MultiTierCache(
            redis_host=config.get('redis_host', 'localhost'),
            redis_port=config.get('redis_port', 6379)
        )
        self.column_mapper = ColumnMapper()
        self.validator = DataValidator()
        
        # Initialize data sources in priority order
        self.sources = []
        
        # File source (primary)
        if file_path := config.get('file_path'):
            self.sources.append(FileDataSource(file_path))
        
        # API source (secondary)
        if api_url := config.get('api_url'):
            self.sources.append(APIDataSource(
                api_url, 
                config.get('api_key')
            ))
        
        # Database source (tertiary)
        if db_conn := config.get('db_connection'):
            self.sources.append(DatabaseDataSource(db_conn))
        
        # Thread pool for parallel loading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache configuration
        self.cache_ttl = {
            'yarn_inventory': 900,  # 15 minutes
            'bom_data': 3600,  # 1 hour
            'production_orders': 60,  # 1 minute
            'work_centers': 3600,  # 1 hour
            'machine_report': 3600,  # 1 hour
            'sales_activity': 1800,  # 30 minutes
            'demand_data': 900  # 15 minutes
        }
    
    async def load_yarn_inventory(self) -> pd.DataFrame:
        """Load yarn inventory with fallback strategy."""
        return await self._load_with_fallback('yarn_inventory')
    
    async def load_bom_data(self) -> pd.DataFrame:
        """Load BOM data with fallback strategy."""
        return await self._load_with_fallback('bom_data')
    
    async def load_production_orders(self) -> pd.DataFrame:
        """Load production orders with fallback strategy."""
        return await self._load_with_fallback('production_orders')
    
    async def load_work_centers(self) -> pd.DataFrame:
        """Load work center data with fallback strategy."""
        return await self._load_with_fallback('work_centers')
    
    async def load_machine_report(self) -> pd.DataFrame:
        """Load machine report with fallback strategy."""
        return await self._load_with_fallback('machine_report')
    
    async def load_sales_activity(self) -> pd.DataFrame:
        """Load sales activity data with fallback strategy."""
        return await self._load_with_fallback('sales_activity')
    
    async def load_demand_data(self) -> pd.DataFrame:
        """Load demand data with fallback strategy."""
        return await self._load_with_fallback('demand_data')
    
    async def _load_with_fallback(self, data_type: str) -> pd.DataFrame:
        """Load data with fallback strategy across sources."""
        # Check cache first
        cache_key = f"data:{data_type}"
        cached_data = await self.cache.get(cache_key, data_type)
        
        if cached_data is not None:
            self.logger.debug(f"Cache hit for {data_type}")
            return cached_data
        
        # Try each source in order
        last_error = None
        for source in self.sources:
            if not source.is_available():
                continue
            
            try:
                self.logger.info(f"Loading {data_type} from {source.__class__.__name__}")
                
                # Load data
                data = await source.load(data_type)
                
                if data is not None and not data.empty:
                    # Standardize columns
                    data = self.column_mapper.standardize(data)
                    
                    # Validate data
                    validation_result = self.validator.validate(data, data_type)
                    if not validation_result.is_valid:
                        self.logger.warning(f"Validation warnings for {data_type}: {validation_result.warnings}")
                    
                    # Handle data type conversions
                    data = self._convert_data_types(data, data_type)
                    
                    # Cache the result
                    ttl = self.cache_ttl.get(data_type, 600)
                    await self.cache.set(cache_key, data, data_type, ttl)
                    
                    self.logger.info(f"Successfully loaded {len(data)} rows for {data_type}")
                    return data
                    
            except Exception as e:
                self.logger.warning(f"Failed to load from {source.__class__.__name__}: {e}")
                last_error = e
                continue
        
        # All sources failed
        raise DataLoadException(f"All data sources failed for {data_type}. Last error: {last_error}")
    
    def _convert_data_types(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Ensure correct data types for specific data."""
        # Common numeric columns
        numeric_columns = [
            'theoretical_balance', 'allocated', 'on_order', 'planning_balance',
            'quantity', 'qty_ordered', 'qty_produced', 'balance',
            'cost', 'price', 'total_cost', 'unit_price'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and dollar signs
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '')
                # Convert to float
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Date columns
        date_columns = [
            'date', 'ship_date', 'due_date', 'created_at', 'updated_at',
            'scheduled_date', 'completion_date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Specific conversions by data type
        if data_type == 'yarn_inventory':
            # Ensure allocated is negative
            if 'allocated' in df.columns:
                df['allocated'] = df['allocated'].abs() * -1
            
            # Calculate planning balance if not present
            if 'planning_balance' not in df.columns:
                if all(col in df.columns for col in ['theoretical_balance', 'allocated', 'on_order']):
                    df['planning_balance'] = (
                        df['theoretical_balance'] + 
                        df['allocated'] + 
                        df['on_order']
                    )
        
        elif data_type == 'production_orders':
            # Ensure status is string
            if 'status' in df.columns:
                df['status'] = df['status'].astype(str).str.lower()
            
            # Calculate completion percentage
            if 'qty_ordered' in df.columns and 'qty_produced' in df.columns:
                df['completion_pct'] = (
                    df['qty_produced'] / df['qty_ordered'].replace(0, np.nan) * 100
                ).fillna(0)
        
        return df
    
    async def load_all_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources in parallel."""
        data_types = [
            'yarn_inventory',
            'bom_data',
            'production_orders',
            'work_centers',
            'machine_report',
            'sales_activity',
            'demand_data'
        ]
        
        # Create loading tasks
        tasks = []
        for data_type in data_types:
            tasks.append(self._load_with_fallback(data_type))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        loaded_data = {}
        for data_type, result in zip(data_types, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to load {data_type}: {result}")
            else:
                loaded_data[data_type] = result
        
        return loaded_data
    
    async def refresh_cache(self, data_type: str = None):
        """Force refresh of cached data."""
        if data_type:
            # Refresh specific data type
            cache_key = f"data:{data_type}"
            await self.cache.delete(cache_key, data_type)
            return await self._load_with_fallback(data_type)
        else:
            # Refresh all
            data_types = ['yarn_inventory', 'bom_data', 'production_orders']
            for dt in data_types:
                cache_key = f"data:{dt}"
                await self.cache.delete(cache_key, dt)
            return await self.load_all_data_sources()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        self.cache.close()