#!/usr/bin/env python3
"""
Unified Data Loader for Beverly Knits ERP
Combines best features from all three existing loaders:
- Parallel processing from parallel_data_loader
- Caching from optimized_data_loader  
- Error handling from erp_data_loader
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    Unified data loader combining best features from all implementations.
    
    Features:
    - Parallel loading for performance
    - Intelligent caching with TTL
    - Robust error handling
    - Column standardization
    - Planning Balance formula fixes
    """
    
    def __init__(self, data_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize unified data loader"""
        
        # Data paths - prioritize actual data location
        self.data_paths = [
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data"),  # Primary location
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-09"),  # Most recent
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-06"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-04"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"),
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data/archive/08-04"),
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data")
        ]
        
        if data_path:
            self.data_paths.insert(0, Path(data_path))
        
        # Find first valid data path
        self.data_path = None
        for path in self.data_paths:
            if path.exists():
                self.data_path = path
                logger.info(f"Using data path: {self.data_path}")
                break
        
        if not self.data_path:
            raise ValueError("No valid data path found")
        
        # Cache configuration
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/bki_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache TTL settings (from optimized_data_loader)
        self.cache_ttl = {
            'yarn_inventory': 900,      # 15 minutes
            'sales_orders': 300,         # 5 minutes
            'bom': 3600,                # 1 hour
            'styles': 1800,              # 30 minutes
            'knit_orders': 600,          # 10 minutes
            'default': 600               # 10 minutes default
        }
        
        # Performance metrics
        self.load_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread pool for parallel loading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Unified Data Loader initialized")
    
    def _get_cache_key(self, file_type: str) -> str:
        """Generate cache key for file type"""
        return f"{file_type}_{self.data_path.name}"
    
    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """Check if cache file is still valid"""
        if not cache_file.exists():
            return False
        
        age = time.time() - cache_file.stat().st_mtime
        return age < ttl
    
    def _load_from_cache(self, file_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_key = self._get_cache_key(file_type)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        ttl = self.cache_ttl.get(file_type, self.cache_ttl['default'])
        
        if self._is_cache_valid(cache_file, ttl):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.cache_hits += 1
                logger.debug(f"Cache hit for {file_type}")
                return data
            except Exception as e:
                logger.warning(f"Cache read error for {file_type}: {e}")
        
        self.cache_misses += 1
        return None
    
    def _save_to_cache(self, file_type: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_key = self._get_cache_key(file_type)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached {file_type}")
        except Exception as e:
            logger.warning(f"Cache write error for {file_type}: {e}")
    
    def _standardize_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Standardize column names based on file type"""
        if df is None or df.empty:
            return df
        
        # Common column mappings
        column_mappings = {
            'Planning_Ballance': 'Planning_Balance',  # Fix typo
            'Theoratical_Balance': 'Theoretical_Balance',
            'Yarn_ID': 'Desc#',
            'YarnID': 'Desc#',
            'Yarn ID': 'Desc#',
            'Desc': 'Desc#',
            'Style #': 'Style#',
            'fStyle': 'fStyle#'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        
        # File-specific fixes
        if file_type == 'yarn_inventory':
            # Ensure Planning Balance column exists and is calculated correctly
            if 'Planning_Balance' not in df.columns:
                if all(col in df.columns for col in ['Theoretical_Balance', 'Allocated', 'On_Order']):
                    # CRITICAL: Allocated is ALREADY NEGATIVE in files
                    df['Planning_Balance'] = df['Theoretical_Balance'] + df['Allocated'] + df['On_Order']
                    logger.info("Calculated Planning_Balance from components")
        
        return df
    
    def _load_file(self, file_pattern: str, file_type: str) -> Optional[pd.DataFrame]:
        """Load a single file with error handling"""
        try:
            # Find matching files
            files = list(self.data_path.glob(file_pattern))
            
            if not files:
                logger.warning(f"No files found for pattern: {file_pattern}")
                return None
            
            # Use most recent file
            file_path = max(files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading {file_type} from: {file_path.name}")
            
            # Load based on extension
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Standardize columns
            df = self._standardize_columns(df, file_type)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_type}: {e}")
            return None
    
    def load_yarn_inventory(self) -> Optional[pd.DataFrame]:
        """Load yarn inventory data with caching"""
        start_time = time.time()
        
        # Check cache first
        df = self._load_from_cache('yarn_inventory')
        if df is not None:
            self.load_times['yarn_inventory'] = time.time() - start_time
            return df
        
        # Load from file
        df = self._load_file('yarn_inventory*.xlsx', 'yarn_inventory')
        if df is None:
            df = self._load_file('yarn_inventory*.csv', 'yarn_inventory')
        
        if df is not None:
            # Critical data validation
            required_cols = ['Desc#', 'Planning_Balance']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            
            # Save to cache
            self._save_to_cache('yarn_inventory', df)
        
        self.load_times['yarn_inventory'] = time.time() - start_time
        logger.info(f"Loaded yarn inventory: {len(df) if df is not None else 0} records in {self.load_times['yarn_inventory']:.2f}s")
        
        return df
    
    def load_sales_orders(self) -> Optional[pd.DataFrame]:
        """Load sales order data"""
        start_time = time.time()
        
        # Check cache
        df = self._load_from_cache('sales_orders')
        if df is not None:
            self.load_times['sales_orders'] = time.time() - start_time
            return df
        
        # Try different file patterns
        patterns = [
            'eFab_SO_List*.csv',
            'Sales Activity Report*.csv',
            'sales_orders*.xlsx'
        ]
        
        for pattern in patterns:
            df = self._load_file(pattern, 'sales_orders')
            if df is not None:
                break
        
        if df is not None:
            self._save_to_cache('sales_orders', df)
        
        self.load_times['sales_orders'] = time.time() - start_time
        logger.info(f"Loaded sales orders: {len(df) if df is not None else 0} records in {self.load_times['sales_orders']:.2f}s")
        
        return df
    
    def load_bom(self) -> Optional[pd.DataFrame]:
        """Load Bill of Materials data"""
        start_time = time.time()
        
        # Check cache
        df = self._load_from_cache('bom')
        if df is not None:
            self.load_times['bom'] = time.time() - start_time
            return df
        
        # Try different BOM file patterns
        patterns = [
            'Style_BOM*.csv',
            'BOM_updated*.csv',
            'BOM_Master*.csv',
            'BOM*.xlsx'
        ]
        
        for pattern in patterns:
            df = self._load_file(pattern, 'bom')
            if df is not None:
                break
        
        if df is not None:
            self._save_to_cache('bom', df)
        
        self.load_times['bom'] = time.time() - start_time
        logger.info(f"Loaded BOM: {len(df) if df is not None else 0} records in {self.load_times['bom']:.2f}s")
        
        return df
    
    def load_all_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """Load all data files in parallel (from parallel_data_loader)"""
        start_time = time.time()
        
        data_loaders = {
            'yarn_inventory': self.load_yarn_inventory,
            'sales_orders': self.load_sales_orders,
            'bom': self.load_bom,
            'knit_orders': self.load_knit_orders,
            'styles': self.load_styles
        }
        
        results = {}
        futures = {}
        
        # Submit all loading tasks
        for name, loader in data_loaders.items():
            future = self.executor.submit(loader)
            futures[future] = name
        
        # Collect results as they complete
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                results[name] = None
        
        total_time = time.time() - start_time
        total_records = sum(len(df) if df is not None else 0 for df in results.values())
        
        logger.info(f"Parallel load complete: {total_records} total records in {total_time:.2f}s")
        logger.info(f"Cache performance: {self.cache_hits} hits, {self.cache_misses} misses")
        
        return results
    
    def load_knit_orders(self) -> Optional[pd.DataFrame]:
        """Load knit orders data"""
        df = self._load_from_cache('knit_orders')
        if df is not None:
            return df
        
        df = self._load_file('eFab_Knit_Orders*.xlsx', 'knit_orders')
        if df is None:
            df = self._load_file('knit_orders*.csv', 'knit_orders')
        
        if df is not None:
            self._save_to_cache('knit_orders', df)
        
        return df
    
    def load_styles(self) -> Optional[pd.DataFrame]:
        """Load styles master data"""
        df = self._load_from_cache('styles')
        if df is not None:
            return df
        
        df = self._load_file('eFab_Styles*.xlsx', 'styles')
        if df is None:
            df = self._load_file('styles*.csv', 'styles')
        
        if df is not None:
            self._save_to_cache('styles', df)
        
        return df
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_files = self.cache_dir.glob("*.pkl")
        count = 0
        for file in cache_files:
            file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get loader performance metrics"""
        return {
            'load_times': self.load_times,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_load_time': sum(self.load_times.values()),
            'data_path': str(self.data_path)
        }
    
    def validate_data_integrity(self) -> Dict[str, List[str]]:
        """Validate loaded data for critical issues"""
        issues = {}
        
        # Load and check yarn inventory
        yarn_df = self.load_yarn_inventory()
        if yarn_df is not None:
            yarn_issues = []
            
            # Check Planning Balance
            if 'Planning_Balance' not in yarn_df.columns:
                yarn_issues.append("Missing Planning_Balance column")
            else:
                # Check for calculation errors
                if all(col in yarn_df.columns for col in ['Theoretical_Balance', 'Allocated', 'On_Order']):
                    calc_balance = yarn_df['Theoretical_Balance'] + yarn_df['Allocated'] + yarn_df['On_Order']
                    discrepancies = (yarn_df['Planning_Balance'] - calc_balance).abs() > 0.01
                    if discrepancies.any():
                        yarn_issues.append(f"Planning Balance calculation errors: {discrepancies.sum()} rows")
            
            # Check for required columns
            required = ['Desc#', 'Planning_Balance', 'Theoretical_Balance']
            missing = [col for col in required if col not in yarn_df.columns]
            if missing:
                yarn_issues.append(f"Missing columns: {missing}")
            
            if yarn_issues:
                issues['yarn_inventory'] = yarn_issues
        else:
            issues['yarn_inventory'] = ['Failed to load data']
        
        # Check other critical data
        sales_df = self.load_sales_orders()
        if sales_df is None:
            issues['sales_orders'] = ['Failed to load data']
        
        bom_df = self.load_bom()
        if bom_df is None:
            issues['bom'] = ['Failed to load data']
        
        return issues


def test_unified_loader():
    """Test the unified data loader"""
    print("=" * 80)
    print("Testing Unified Data Loader")
    print("=" * 80)
    
    loader = UnifiedDataLoader()
    
    # Test individual loaders
    print("\n1. Testing individual loaders:")
    
    yarn_df = loader.load_yarn_inventory()
    print(f"   Yarn Inventory: {len(yarn_df) if yarn_df is not None else 'Failed'} records")
    
    sales_df = loader.load_sales_orders()
    print(f"   Sales Orders: {len(sales_df) if sales_df is not None else 'Failed'} records")
    
    bom_df = loader.load_bom()
    print(f"   BOM: {len(bom_df) if bom_df is not None else 'Failed'} records")
    
    # Test parallel loading
    print("\n2. Testing parallel loading:")
    all_data = loader.load_all_data_parallel()
    for name, df in all_data.items():
        print(f"   {name}: {len(df) if df is not None else 'Failed'} records")
    
    # Test data validation
    print("\n3. Validating data integrity:")
    issues = loader.validate_data_integrity()
    if issues:
        print("   Issues found:")
        for data_type, issue_list in issues.items():
            for issue in issue_list:
                print(f"     - {data_type}: {issue}")
    else:
        print("   ✅ No issues found")
    
    # Performance metrics
    print("\n4. Performance Metrics:")
    metrics = loader.get_performance_metrics()
    print(f"   Total load time: {metrics['total_load_time']:.2f}s")
    print(f"   Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
    print(f"   Data path: {metrics['data_path']}")
    
    print("\n" + "=" * 80)
    print("✅ Unified Data Loader test complete")
    

if __name__ == "__main__":
    test_unified_loader()