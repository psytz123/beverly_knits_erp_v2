"""
eFab.ai API Data Loader
API-first data loader with intelligent fallback to file-based loading
Extends ConsolidatedDataLoader for seamless integration
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.data_loaders.consolidated_data_loader import ConsolidatedDataLoader
    from src.utils.column_standardizer import ColumnStandardizer
except ImportError:
    # Fallback imports for different execution contexts
    try:
        from data_loaders.consolidated_data_loader import ConsolidatedDataLoader
        from utils.column_standardizer import ColumnStandardizer
    except ImportError:
        ConsolidatedDataLoader = object
        ColumnStandardizer = None

from src.api_clients.efab_api_client import EFabAPIClient, CircuitOpenError
from src.api_clients.efab_transformers import EFabDataTransformer
from src.config.secure_api_config import get_api_config

logger = logging.getLogger(__name__)


class EFabAPIDataLoader(ConsolidatedDataLoader if ConsolidatedDataLoader != object else object):
    """
    API-first data loader with intelligent fallback to file-based loading
    Seamlessly integrates with existing ERP infrastructure
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize API data loader
        
        Args:
            data_path: Optional override for data path
        """
        # Initialize parent if available
        if ConsolidatedDataLoader != object:
            super().__init__(data_path)
        else:
            self.data_path = data_path or "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data"
            self.cache = {}
            self.last_load_time = {}
        
        # Get API configuration
        self.config_manager = get_api_config()
        self.config = self.config_manager.get_credentials()
        self.feature_flags = self.config_manager.get_feature_flags()
        
        # Initialize API components if enabled
        self.api_enabled = self.feature_flags.get('api_enabled', False)
        self.api_client = None
        self.transformer = EFabDataTransformer()
        self.api_available = False
        
        # API availability check
        if self.api_enabled:
            self._init_api_client()
        
        # Fallback settings
        self.enable_fallback = self.feature_flags.get('enable_fallback', True)
        self.api_failure_count = 0
        self.max_api_failures = 3
        
        logger.info(f"EFabAPIDataLoader initialized - API: {self.api_enabled}, Fallback: {self.enable_fallback}")
    
    def _init_api_client(self):
        """Initialize API client and check availability"""
        try:
            self.api_client = EFabAPIClient(self.config)
            # Note: Actual availability check should be async, handled in load methods
            self.api_available = True
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize API client: {e}")
            self.api_available = False
    
    async def _check_api_availability(self) -> bool:
        """
        Check if API is available and healthy
        
        Returns:
            True if API is available
        """
        if not self.api_enabled or not self.api_client:
            return False
        
        try:
            # Initialize client if needed
            if not hasattr(self.api_client, '_session') or not self.api_client._session:
                await self.api_client.initialize()
            
            # Perform health check
            is_healthy = await self.api_client.health_check()
            
            if is_healthy:
                self.api_failure_count = 0
                self.api_available = True
            else:
                self.api_failure_count += 1
                self.api_available = False
            
            return is_healthy
            
        except CircuitOpenError:
            logger.warning("Circuit breaker is open, API unavailable")
            self.api_available = False
            return False
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            self.api_failure_count += 1
            self.api_available = False
            return False
    
    def load_yarn_inventory(self) -> pd.DataFrame:
        """
        Load yarn inventory with API-first strategy
        
        Returns:
            Yarn inventory DataFrame
        """
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_yarn_inventory_async())
        finally:
            loop.close()
    
    async def _load_yarn_inventory_async(self) -> pd.DataFrame:
        """
        Async yarn inventory loading
        
        Returns:
            Yarn inventory DataFrame
        """
        logger.info("Loading yarn inventory...")
        
        # Try API first if enabled
        if self.api_enabled and await self._check_api_availability():
            try:
                logger.info("Attempting to load yarn inventory from API...")
                
                # Get data from API
                api_data = await self.api_client.get_yarn_active()
                
                # Transform to ERP format
                df = self.transformer.transform_yarn_active(api_data)
                
                # Validate transformation
                if self.transformer.validate_transformation(api_data, df, 'yarn_inventory'):
                    # Standardize columns if available
                    if ColumnStandardizer:
                        df = ColumnStandardizer.standardize_dataframe(df, 'yarn_inventory')
                    
                    # Cache the result
                    self.cache['yarn_inventory'] = df
                    self.last_load_time['yarn_inventory'] = datetime.now()
                    
                    logger.info(f"Successfully loaded {len(df)} yarn inventory records from API")
                    return df
                else:
                    logger.warning("API data validation failed, falling back to files")
                    
            except Exception as e:
                logger.error(f"API load failed for yarn inventory: {e}")
                self.api_failure_count += 1
        
        # Fallback to file loading
        if self.enable_fallback:
            logger.info("Loading yarn inventory from files...")
            
            if hasattr(super(), 'load_yarn_inventory'):
                return super().load_yarn_inventory()
            else:
                # Manual file loading
                return self._load_yarn_inventory_from_file()
        
        logger.error("Failed to load yarn inventory from both API and files")
        return pd.DataFrame()
    
    def _load_yarn_inventory_from_file(self) -> pd.DataFrame:
        """
        Fallback method to load yarn inventory from file
        
        Returns:
            Yarn inventory DataFrame
        """
        try:
            file_path = Path(self.data_path) / "yarn_inventory.xlsx"
            
            if not file_path.exists():
                # Try alternate path
                file_path = Path(self.data_path) / "8-28-2025" / "yarn_inventory.xlsx"
            
            if file_path.exists():
                df = pd.read_excel(file_path)
                
                # Calculate Planning Balance if needed
                if 'Planning Balance' not in df.columns and 'Planning_Balance' not in df.columns:
                    if all(col in df.columns for col in ['Theoretical Balance', 'Allocated', 'On Order']):
                        df['Planning Balance'] = (
                            df['Theoretical Balance'] +
                            df['Allocated'] +  # Already negative
                            df['On Order']
                        )
                
                logger.info(f"Loaded {len(df)} yarn inventory records from file")
                return df
            else:
                logger.error(f"Yarn inventory file not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading yarn inventory from file: {e}")
            return pd.DataFrame()
    
    def load_knit_orders(self) -> pd.DataFrame:
        """
        Load knit orders with API-first strategy
        
        Returns:
            Knit orders DataFrame
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_knit_orders_async())
        finally:
            loop.close()
    
    async def _load_knit_orders_async(self) -> pd.DataFrame:
        """
        Async knit orders loading
        
        Returns:
            Knit orders DataFrame
        """
        logger.info("Loading knit orders...")
        
        # Try API first
        if self.api_enabled and await self._check_api_availability():
            try:
                logger.info("Attempting to load knit orders from API...")
                
                # Get data from API
                api_data = await self.api_client.get_knit_orders()
                
                # Transform to ERP format
                df = self.transformer.transform_knit_orders(api_data)
                
                # Validate and cache
                if not df.empty:
                    self.cache['knit_orders'] = df
                    self.last_load_time['knit_orders'] = datetime.now()
                    
                    logger.info(f"Successfully loaded {len(df)} knit orders from API")
                    return df
                    
            except Exception as e:
                logger.error(f"API load failed for knit orders: {e}")
        
        # Fallback to file loading
        if self.enable_fallback:
            logger.info("Loading knit orders from files...")
            
            if hasattr(super(), 'load_efab_knit_orders'):
                return super().load_efab_knit_orders()
            else:
                return self._load_knit_orders_from_file()
        
        return pd.DataFrame()
    
    def _load_knit_orders_from_file(self) -> pd.DataFrame:
        """
        Load knit orders from file
        
        Returns:
            Knit orders DataFrame
        """
        try:
            file_path = Path(self.data_path) / "eFab_Knit_Orders.csv"
            
            if not file_path.exists():
                file_path = Path(self.data_path) / "8-28-2025" / "eFab_Knit_Orders.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} knit orders from file")
                return df
            else:
                logger.error(f"Knit orders file not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading knit orders from file: {e}")
            return pd.DataFrame()
    
    def load_po_deliveries(self) -> pd.DataFrame:
        """
        Load PO deliveries (time-phased) with API-first strategy
        
        Returns:
            PO deliveries DataFrame with weekly buckets
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_po_deliveries_async())
        finally:
            loop.close()
    
    async def _load_po_deliveries_async(self) -> pd.DataFrame:
        """
        Async PO deliveries loading
        
        Returns:
            PO deliveries DataFrame
        """
        logger.info("Loading PO deliveries...")
        
        # Try API first
        if self.api_enabled and await self._check_api_availability():
            try:
                logger.info("Attempting to load PO deliveries from API...")
                
                # Get data from API
                api_data = await self.api_client.get_yarn_expected()
                
                # Transform to time-phased format
                df = self.transformer.transform_yarn_expected(api_data)
                
                if not df.empty:
                    self.cache['po_deliveries'] = df
                    self.last_load_time['po_deliveries'] = datetime.now()
                    
                    logger.info(f"Successfully loaded {len(df)} PO delivery records from API")
                    return df
                    
            except Exception as e:
                logger.error(f"API load failed for PO deliveries: {e}")
        
        # Fallback to file loading
        if self.enable_fallback:
            logger.info("Loading PO deliveries from files...")
            
            if hasattr(super(), 'load_po_deliveries'):
                return super().load_po_deliveries()
            else:
                return self._load_po_deliveries_from_file()
        
        return pd.DataFrame()
    
    def _load_po_deliveries_from_file(self) -> pd.DataFrame:
        """
        Load PO deliveries from file
        
        Returns:
            PO deliveries DataFrame
        """
        try:
            file_path = Path(self.data_path) / "Expected_Yarn_Report.xlsx"
            
            if not file_path.exists():
                file_path = Path(self.data_path) / "8-28-2025" / "Expected_Yarn_Report.xlsx"
            
            if file_path.exists():
                df = pd.read_excel(file_path)
                logger.info(f"Loaded {len(df)} PO delivery records from file")
                return df
            else:
                logger.error(f"PO deliveries file not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading PO deliveries from file: {e}")
            return pd.DataFrame()
    
    def load_sales_activity(self) -> pd.DataFrame:
        """
        Load sales activity with API-first strategy
        
        Returns:
            Sales activity DataFrame
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_sales_activity_async())
        finally:
            loop.close()
    
    async def _load_sales_activity_async(self) -> pd.DataFrame:
        """
        Async sales activity loading
        
        Returns:
            Sales activity DataFrame
        """
        logger.info("Loading sales activity...")
        
        # Try API first
        if self.api_enabled and await self._check_api_availability():
            try:
                logger.info("Attempting to load sales activity from API...")
                
                # Get data from API
                api_data = await self.api_client.get_sales_activity()
                
                # Transform to ERP format
                df = self.transformer.transform_sales_activity(api_data)
                
                if not df.empty:
                    self.cache['sales_activity'] = df
                    self.last_load_time['sales_activity'] = datetime.now()
                    
                    logger.info(f"Successfully loaded {len(df)} sales activity records from API")
                    return df
                    
            except Exception as e:
                logger.error(f"API load failed for sales activity: {e}")
        
        # Fallback to file loading
        if self.enable_fallback:
            logger.info("Loading sales activity from files...")
            
            if hasattr(super(), 'load_sales_data'):
                return super().load_sales_data()
            else:
                return self._load_sales_activity_from_file()
        
        return pd.DataFrame()
    
    def _load_sales_activity_from_file(self) -> pd.DataFrame:
        """
        Load sales activity from file
        
        Returns:
            Sales activity DataFrame
        """
        try:
            file_path = Path(self.data_path) / "Sales Activity Report.csv"
            
            if not file_path.exists():
                file_path = Path(self.data_path) / "8-28-2025" / "Sales Activity Report.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} sales activity records from file")
                return df
            else:
                logger.error(f"Sales activity file not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading sales activity from file: {e}")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data with API-first strategy
        
        Returns:
            Dictionary mapping data type to DataFrame
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_all_data_async())
        finally:
            loop.close()
    
    async def _load_all_data_async(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data asynchronously
        
        Returns:
            Dictionary of DataFrames
        """
        logger.info("Loading all data...")
        
        # Try parallel API loading first
        if self.api_enabled and await self._check_api_availability():
            try:
                logger.info("Attempting parallel API data loading...")
                
                # Get all data in parallel
                all_data = await self.api_client.get_all_data_parallel()
                
                # Transform each dataset
                transformed_data = {}
                
                if 'yarn_inventory' in all_data:
                    transformed_data['yarn_inventory'] = self.transformer.transform_yarn_active(all_data['yarn_inventory'])
                
                if 'knit_orders' in all_data:
                    transformed_data['knit_orders'] = self.transformer.transform_knit_orders(all_data['knit_orders'])
                
                if 'yarn_expected' in all_data:
                    transformed_data['po_deliveries'] = self.transformer.transform_yarn_expected(all_data['yarn_expected'])
                
                if 'sales_activity' in all_data:
                    transformed_data['sales_activity'] = self.transformer.transform_sales_activity(all_data['sales_activity'])
                
                if 'greige_g00' in all_data:
                    transformed_data['greige_g00'] = self.transformer.transform_greige_inventory(all_data['greige_g00'], 'g00')
                
                if 'greige_g02' in all_data:
                    transformed_data['greige_g02'] = self.transformer.transform_greige_inventory(all_data['greige_g02'], 'g02')
                
                # Add other data as-is
                for key in ['finished_i01', 'finished_f01', 'styles', 'yarn_demand', 'sales_order_plan']:
                    if key in all_data:
                        transformed_data[key] = all_data[key]
                
                # Cache all data
                self.cache.update(transformed_data)
                for key in transformed_data:
                    self.last_load_time[key] = datetime.now()
                
                logger.info(f"Successfully loaded {len(transformed_data)} datasets from API")
                return transformed_data
                
            except Exception as e:
                logger.error(f"Parallel API load failed: {e}")
        
        # Fallback to sequential loading
        logger.info("Loading data sequentially...")
        
        data = {}
        data['yarn_inventory'] = self.load_yarn_inventory()
        data['knit_orders'] = self.load_knit_orders()
        data['po_deliveries'] = self.load_po_deliveries()
        data['sales_activity'] = self.load_sales_activity()
        
        # Load other data from parent if available
        if hasattr(super(), 'load_bom'):
            data['bom'] = super().load_bom()
        
        if hasattr(super(), 'load_machine_report'):
            data['machine_report'] = super().load_machine_report()
        
        return data
    
    def get_loader_status(self) -> Dict[str, Any]:
        """
        Get loader status and statistics
        
        Returns:
            Status dictionary
        """
        status = {
            'api_enabled': self.api_enabled,
            'api_available': self.api_available,
            'fallback_enabled': self.enable_fallback,
            'api_failure_count': self.api_failure_count,
            'cached_datasets': list(self.cache.keys()),
            'last_load_times': {
                k: v.isoformat() if v else None
                for k, v in self.last_load_time.items()
            }
        }
        
        if self.api_client:
            status['api_client_status'] = self.api_client.get_status()
        
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.last_load_time.clear()
        
        if self.api_client and self.api_client.cache:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.api_client.cache.clear())
            finally:
                loop.close()
        
        logger.info("All caches cleared")


def test_api_loader():
    """Test API data loader"""
    print("Testing EFab API Data Loader")
    print("=" * 50)
    
    # Create loader
    loader = EFabAPIDataLoader()
    
    # Get status
    print("\nLoader Status:")
    status = loader.get_loader_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test loading yarn inventory
    print("\nLoading Yarn Inventory:")
    yarn_df = loader.load_yarn_inventory()
    print(f"  Loaded {len(yarn_df)} records")
    if not yarn_df.empty:
        print(f"  Columns: {list(yarn_df.columns)[:5]}...")
    
    # Test loading knit orders
    print("\nLoading Knit Orders:")
    orders_df = loader.load_knit_orders()
    print(f"  Loaded {len(orders_df)} records")
    
    # Test loading all data
    print("\nLoading All Data:")
    all_data = loader.load_all_data()
    for name, df in all_data.items():
        print(f"  {name}: {len(df)} records")


if __name__ == "__main__":
    test_api_loader()