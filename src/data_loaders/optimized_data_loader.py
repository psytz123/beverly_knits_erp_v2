#!/usr/bin/env python3
"""
Optimized Data Loading Module for Beverly Knits ERP
Implements batch loading, caching, and parallel processing for improved performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pickle
import hashlib
import os
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class OptimizedDataLoader:
    """
    Optimized data loading with caching, batch processing, and parallel loading.
    Improves performance for Beverly Knits ERP data operations.
    """
    
    def __init__(self, data_path: Path, cache_dir: str = None):
        """
        Initialize the optimized data loader.
        
        Args:
            data_path: Base path to ERP data files
            cache_dir: Directory for cached data (defaults to temp dir)
        """
        self.data_path = Path(data_path)
        
        # Use platform-appropriate temp directory
        if cache_dir is None:
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), 'bki_cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache TTL settings (in minutes)
        self.cache_ttl = {
            'yarn_inventory': 15,      # Refresh every 15 minutes
            'sales_orders': 5,          # Refresh every 5 minutes
            'bom_data': 60,            # Refresh every hour
            'style_mappings': 1440,    # Refresh daily
            'inventory_f01': 10,       # Refresh every 10 minutes
            'inventory_g00': 10,       
            'inventory_g02': 10,
            'inventory_i01': 10,
            'knit_orders': 5           # Refresh every 5 minutes
        }
        
        # In-memory cache for frequently accessed data
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'loads': 0
        }
        
    def get_cache_key(self, category: str, params: Dict = None) -> str:
        """Generate a unique cache key for data category and parameters."""
        params = params or {}
        param_str = str(sorted(params.items()))
        hash_val = hashlib.md5(f"{category}_{param_str}".encode()).hexdigest()
        return f"{category}_{hash_val}"
    
    def is_cache_valid(self, cache_file: Path, ttl_minutes: int) -> bool:
        """Check if a cache file is still valid based on TTL."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(minutes=ttl_minutes)
    
    def save_to_cache(self, data: Any, category: str, params: Dict = None):
        """Save data to cache file."""
        cache_key = self.get_cache_key(category, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now(),
                    'category': category,
                    'params': params
                }, f)
            logger.debug(f"Cached {category} to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache {category}: {e}")
    
    def load_from_cache(self, category: str, params: Dict = None) -> Optional[Any]:
        """Load data from cache if valid."""
        cache_key = self.get_cache_key(category, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        ttl = self.cache_ttl.get(category, 30)
        
        if self.is_cache_valid(cache_file, ttl):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.cache_stats['hits'] += 1
                    logger.debug(f"Cache hit for {category}")
                    return cached_data['data']
            except Exception as e:
                logger.error(f"Failed to load cache for {category}: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    @lru_cache(maxsize=128)
    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load Excel file with in-memory caching."""
        return pd.read_excel(file_path)
    
    @lru_cache(maxsize=128)
    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with in-memory caching."""
        return pd.read_csv(file_path)
    
    def batch_load_inventory(self) -> Dict[str, pd.DataFrame]:
        """
        Load all inventory files in parallel using ThreadPoolExecutor.
        Returns dict with stage names as keys and DataFrames as values.
        """
        # Check cache first
        cached_data = self.load_from_cache('all_inventory')
        if cached_data is not None:
            return cached_data
        
        inventory_stages = ['F01', 'G00', 'G02', 'I01']
        inventory_data = {}
        
        def load_stage(stage):
            """Load a single inventory stage."""
            primary_path = self.data_path / "prompts" / "5"
            if not primary_path.exists():
                primary_path = self.data_path / "5"
            
            # Look for files matching the stage pattern
            patterns = [
                f"eFab_Inventory_{stage}_*.xlsx",
                f"eFab_Inventory_{stage}_*.csv"
            ]
            
            for pattern in patterns:
                files = list(primary_path.glob(pattern))
                if files:
                    # Use the most recent file
                    latest_file = sorted(files)[-1]
                    try:
                        if latest_file.suffix == '.xlsx':
                            data = pd.read_excel(latest_file)
                        else:
                            data = pd.read_csv(latest_file)
                        
                        return stage, data
                    except Exception as e:
                        logger.error(f"Error loading {stage}: {e}")
                        return stage, pd.DataFrame()
            
            return stage, pd.DataFrame()
        
        # Load all stages in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_stage, stage) for stage in inventory_stages]
            
            for future in as_completed(futures):
                stage, data = future.result()
                if not data.empty:
                    inventory_data[stage] = data
                    logger.info(f"Loaded {stage}: {len(data)} records")
        
        # Save to cache
        self.save_to_cache(inventory_data, 'all_inventory')
        self.cache_stats['loads'] += len(inventory_data)
        
        return inventory_data
    
    def load_yarn_inventory(self) -> pd.DataFrame:
        """Load yarn inventory with caching."""
        # Check cache
        cached_data = self.load_from_cache('yarn_inventory')
        if cached_data is not None:
            return cached_data
        
        # Load fresh data - check multiple possible paths
        possible_paths = [
            self.data_path,  # Direct path (e.g., ERP Data/5)
            self.data_path / "5",  # In case path is ERP Data
            self.data_path / "prompts" / "5"  # Legacy path
        ]
        
        for primary_path in possible_paths:
            if not primary_path.exists():
                continue
                
            # First priority: yarn_inventory (4) files
            yarn_4_xlsx = primary_path / "yarn_inventory (4).xlsx"
            yarn_4_csv = primary_path / "yarn_inventory (4).csv"
            
            if yarn_4_xlsx.exists():
                logger.info(f"Loading yarn_inventory (4).xlsx from {primary_path}")
                data = pd.read_excel(yarn_4_xlsx)
                self.save_to_cache(data, 'yarn_inventory')
                self.cache_stats['loads'] += 1
                return data
            elif yarn_4_csv.exists():
                logger.info(f"Loading yarn_inventory (4).csv from {primary_path}")
                data = pd.read_csv(yarn_4_csv)
                self.save_to_cache(data, 'yarn_inventory')
                self.cache_stats['loads'] += 1
                return data
            
            # Fallback: Try to find the latest yarn_inventory file
            yarn_files = list(primary_path.glob("yarn_inventory*.xlsx"))
            if yarn_files:
                # Get the latest yarn inventory file
                yarn_file = sorted(yarn_files)[-1]
                logger.info(f"Loading {yarn_file.name} from {primary_path}")
                data = pd.read_excel(yarn_file)
                self.save_to_cache(data, 'yarn_inventory')
                self.cache_stats['loads'] += 1
                return data
        
        # Final fallback to other yarn files
        for primary_path in possible_paths:
            if not primary_path.exists():
                continue
                
            yarn_patterns = ["Yarn_ID*.csv", "yarn_*.xlsx"]
            for pattern in yarn_patterns:
                files = list(primary_path.glob(pattern))
                if files:
                    latest_file = sorted(files)[-1]
                    logger.info(f"Loading fallback {latest_file.name} from {primary_path}")
                    if latest_file.suffix == '.xlsx':
                        data = pd.read_excel(latest_file)
                    else:
                        data = pd.read_csv(latest_file)
                    
                    self.save_to_cache(data, 'yarn_inventory')
                    self.cache_stats['loads'] += 1
                    return data
        
        logger.warning("No yarn inventory file found")
        return pd.DataFrame()
    
    def load_sales_orders(self) -> pd.DataFrame:
        """Load sales orders with caching."""
        # Check cache
        cached_data = self.load_from_cache('sales_orders')
        if cached_data is not None:
            return cached_data
        
        primary_path = self.data_path / "prompts" / "5"
        if not primary_path.exists():
            primary_path = self.data_path / "5"
        
        # Look for SO files
        so_patterns = [
            "eFab_SO_List*.csv",
            "eFab_SO_List*.xlsx",
            "Sales Activity Report*.csv",
            "Sales Activity Report*.xlsx"
        ]
        
        for pattern in so_patterns:
            files = list(primary_path.glob(pattern))
            if files:
                latest_file = sorted(files)[-1]
                try:
                    if latest_file.suffix == '.xlsx':
                        data = pd.read_excel(latest_file)
                    else:
                        data = pd.read_csv(latest_file)
                    
                    self.save_to_cache(data, 'sales_orders')
                    self.cache_stats['loads'] += 1
                    return data
                except Exception as e:
                    logger.error(f"Error loading sales orders: {e}")
        
        return pd.DataFrame()
    
    def load_bom_data(self) -> pd.DataFrame:
        """Load BOM data with caching."""
        # Check cache
        cached_data = self.load_from_cache('bom_data')
        if cached_data is not None:
            return cached_data
        
        primary_path = self.data_path / "prompts" / "5"
        if not primary_path.exists():
            primary_path = self.data_path / "5"
        
        # BOM file patterns in priority order
        bom_patterns = [
            "BOM_updated.csv",
            "Style_BOM copy.csv",
            "Style_BOM.csv",
            "BOM*.csv"
        ]
        
        for pattern in bom_patterns:
            files = list(primary_path.glob(pattern))
            if files:
                bom_file = files[0]
                try:
                    data = pd.read_csv(bom_file)
                    self.save_to_cache(data, 'bom_data')
                    self.cache_stats['loads'] += 1
                    return data
                except Exception as e:
                    logger.error(f"Error loading BOM: {e}")
        
        return pd.DataFrame()
    
    def load_knit_orders(self) -> pd.DataFrame:
        """Load knit orders with caching."""
        # Check cache
        cached_data = self.load_from_cache('knit_orders')
        if cached_data is not None:
            return cached_data
        
        primary_path = self.data_path / "prompts" / "5"
        if not primary_path.exists():
            primary_path = self.data_path / "5"
        
        ko_file = primary_path / "eFab_Knit_Orders_20250816.xlsx"
        
        if ko_file.exists():
            try:
                data = pd.read_excel(ko_file)
                self.save_to_cache(data, 'knit_orders')
                self.cache_stats['loads'] += 1
                return data
            except Exception as e:
                logger.error(f"Error loading knit orders: {e}")
        
        return pd.DataFrame()
    
    def load_all_data_optimized(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources in parallel with caching.
        Returns a dictionary with all loaded DataFrames.
        """
        logger.info("Starting optimized data loading...")
        start_time = datetime.now()
        
        all_data = {}
        
        # Define loading tasks
        loading_tasks = [
            ('yarn_inventory', self.load_yarn_inventory),
            ('sales_orders', self.load_sales_orders),
            ('bom_data', self.load_bom_data),
            ('knit_orders', self.load_knit_orders),
            ('inventory', self.batch_load_inventory)
        ]
        
        # Load in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(loader): name 
                for name, loader in loading_tasks
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_data[name] = result
                    
                    if isinstance(result, pd.DataFrame):
                        logger.info(f"Loaded {name}: {len(result)} records")
                    elif isinstance(result, dict):
                        logger.info(f"Loaded {name}: {len(result)} stages")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
                    all_data[name] = pd.DataFrame() if name != 'inventory' else {}
        
        # Calculate and log performance metrics
        load_time = (datetime.now() - start_time).total_seconds()
        cache_hit_rate = (self.cache_stats['hits'] / 
                         max(1, self.cache_stats['hits'] + self.cache_stats['misses'])) * 100
        
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
        logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(f"Total loads from disk: {self.cache_stats['loads']}")
        
        return all_data
    
    def clear_cache(self, category: Optional[str] = None):
        """
        Clear cache for a specific category or all categories.
        
        Args:
            category: Specific category to clear, or None for all
        """
        if category:
            pattern = f"{category}_*.pkl"
        else:
            pattern = "*.pkl"
        
        cache_files = list(self.cache_dir.glob(pattern))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file.name}")
            except Exception as e:
                logger.error(f"Failed to clear cache {cache_file}: {e}")
        
        # Clear in-memory cache
        if not category:
            self.memory_cache.clear()
            self.load_excel_file.cache_clear()
            self.load_csv_file.cache_clear()
        
        logger.info(f"Cache cleared for: {category or 'all categories'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = self.cache_stats.copy()
        stats['cache_size_mb'] = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        ) / (1024 * 1024)
        stats['cache_files'] = len(list(self.cache_dir.glob("*.pkl")))
        stats['hit_rate'] = (stats['hits'] / 
                            max(1, stats['hits'] + stats['misses'])) * 100
        return stats


def integrate_with_erp(erp_instance):
    """
    Integrate optimized loader with existing ERP instance.
    
    Args:
        erp_instance: Instance of InventoryManagementPipeline or similar
    """
    # Create optimized loader
    loader = OptimizedDataLoader(erp_instance.data_path)
    
    # Replace load methods with optimized versions
    original_load = erp_instance.load_all_data
    
    def optimized_load_all():
        """Replacement for load_all_data using optimized loader."""
        logger.info("Using optimized data loading...")
        
        # Load all data optimized
        all_data = loader.load_all_data_optimized()
        
        # Assign to ERP instance attributes
        erp_instance.raw_materials_data = all_data.get('yarn_inventory', pd.DataFrame())
        erp_instance.yarn_data = erp_instance.raw_materials_data
        erp_instance.sales_data = all_data.get('sales_orders', pd.DataFrame())
        erp_instance.bom_data = all_data.get('bom_data', pd.DataFrame())
        
        # Handle inventory data
        inventory = all_data.get('inventory', {})
        erp_instance.inventory_data = {}
        for stage, data in inventory.items():
            erp_instance.inventory_data[stage] = {
                'data': data,
                'file': f"eFab_Inventory_{stage}_optimized.xlsx",
                'count': len(data),
                'loaded_at': datetime.now()
            }
        
        # Handle knit orders if available
        if hasattr(erp_instance, 'knit_orders'):
            erp_instance.knit_orders = all_data.get('knit_orders', pd.DataFrame())
        
        logger.info(f"Optimized loading complete: {len(all_data)} data sources")
        
    # Replace the method
    erp_instance.load_all_data = optimized_load_all
    
    # Add cache management methods
    erp_instance.clear_data_cache = loader.clear_cache
    erp_instance.get_cache_stats = loader.get_cache_stats
    
    return loader


if __name__ == "__main__":
    # Test the optimized loader
    import sys
    
    data_path = Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data")
    loader = OptimizedDataLoader(data_path)
    
    print("Testing optimized data loader...")
    print("-" * 50)
    
    # Test loading all data
    start = datetime.now()
    all_data = loader.load_all_data_optimized()
    duration = (datetime.now() - start).total_seconds()
    
    print(f"\nLoading completed in {duration:.2f} seconds")
    print("\nData loaded:")
    for name, data in all_data.items():
        if isinstance(data, pd.DataFrame):
            print(f"  {name}: {len(data)} records")
        elif isinstance(data, dict):
            print(f"  {name}: {len(data)} stages")
    
    # Show cache stats
    stats = loader.get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    print(f"  Cache files: {stats['cache_files']}")
    print(f"  Total loads from disk: {stats['loads']}")
    
    # Test second load (should be cached)
    print("\nTesting cached load...")
    start = datetime.now()
    all_data2 = loader.load_all_data_optimized()
    duration2 = (datetime.now() - start).total_seconds()
    
    print(f"Cached loading completed in {duration2:.2f} seconds")
    print(f"Speed improvement: {(duration/duration2):.1f}x faster")
    
    # Updated cache stats
    stats2 = loader.get_cache_stats()
    print(f"Updated hit rate: {stats2['hit_rate']:.1f}%")