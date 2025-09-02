#!/usr/bin/env python3
"""
Consolidated Data Loader for Beverly Knits ERP
Combines ALL features from the 4 existing loaders:
- Advanced caching with TTL from optimized_data_loader
- 5-worker parallel processing from parallel_data_loader  
- Database integration from database_loader
- Column standardization and data validation from unified_data_loader

This is the SINGLE data loader for the entire ERP system.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
from functools import lru_cache
import tempfile
import glob

# Import column standardizer
try:
    from src.utils.column_standardization import ColumnStandardizer
except ImportError:
    # Fallback for different import paths
    try:
        from utils.column_standardization import ColumnStandardizer
    except ImportError:
        logger.warning("ColumnStandardizer not available, column standardization disabled")
        ColumnStandardizer = None

# Database imports (optional)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsolidatedDataLoader:
    """
    Consolidated data loader combining ALL features from existing implementations.
    
    Features:
    - 5-worker parallel loading (from ParallelDataLoader)
    - Advanced caching with TTL and statistics (from OptimizedDataLoader)
    - Database integration with fallback (from DatabaseDataLoader)  
    - Column standardization and data validation (from UnifiedDataLoader)
    - Planning Balance formula corrections
    - ERP system integration helpers
    - Comprehensive error handling and logging
    
    This replaces ALL existing data loaders.
    """
    
    def __init__(self, data_path: Optional[str] = None, cache_dir: Optional[str] = None, max_workers: int = 5):
        """Initialize consolidated data loader with all features"""
        
        # Data paths - comprehensive search from all loaders
        self.data_paths = [
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data"),  # Current primary
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data"),  # Legacy primary
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-09"),  # Most recent
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-06"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-04"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"),
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5"),
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
        
        # Enhanced cache configuration (from OptimizedDataLoader)
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'bki_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache TTL settings (in minutes, from OptimizedDataLoader)
        self.cache_ttl = {
            'yarn_inventory': 15,        # 15 minutes
            'sales_orders': 5,           # 5 minutes  
            'bom_data': 60,             # 1 hour
            'style_mappings': 1440,      # 24 hours
            'inventory_f01': 10,         # 10 minutes
            'inventory_g00': 10,         
            'inventory_g02': 10,
            'inventory_i01': 10,
            'knit_orders': 5,            # 5 minutes
            'styles': 30,                # 30 minutes
            'default': 10                # 10 minutes default
        }
        
        # Performance metrics and cache stats (from OptimizedDataLoader)
        self.load_times = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0, 
            'loads': 0
        }
        
        # Enhanced thread pool (5 workers from ParallelDataLoader)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # In-memory cache for frequently accessed data
        self.memory_cache = {}
        
        # Database configuration (from DatabaseDataLoader)
        self.use_database = False
        self.db_conn = None
        self.db_config = None
        
        # Try to load database configuration if available
        self._init_database_config()
        
        logger.info(f"Consolidated Data Loader initialized - {self.max_workers} workers, caching enabled")
    
    def _init_database_config(self):
        """Initialize database configuration (from DatabaseDataLoader)"""
        try:
            config_path = Path(__file__).parent.parent / 'config' / 'unified_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.db_config = config.get('data_source', {}).get('database')
                    self.use_database = config.get('data_source', {}).get('primary') == 'database'
                    if self.use_database and DATABASE_AVAILABLE:
                        logger.info("Database configuration loaded")
        except Exception as e:
            logger.debug(f"No database config found: {e}")
            self.use_database = False
    
    def _connect_database(self):
        """Establish database connection (from DatabaseDataLoader)"""
        if not self.use_database or not DATABASE_AVAILABLE or self.db_conn:
            return
            
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'], 
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Using file loading.")
            self.use_database = False
    
    def get_cache_key(self, category: str, params: Dict = None) -> str:
        """Generate unique cache key (from OptimizedDataLoader)"""
        params = params or {}
        param_str = str(sorted(params.items()))
        hash_val = hashlib.md5(f"{category}_{param_str}".encode()).hexdigest()
        return f"{category}_{hash_val}"
    
    def is_cache_valid(self, cache_file: Path, ttl_minutes: int) -> bool:
        """Check if cache file is valid (from OptimizedDataLoader)"""
        if not cache_file.exists():
            return False
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(minutes=ttl_minutes)
    
    def save_to_cache(self, data: Any, category: str, params: Dict = None):
        """Save data to cache (from OptimizedDataLoader)"""
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
        """Load data from cache if valid (from OptimizedDataLoader)"""
        cache_key = self.get_cache_key(category, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        ttl = self.cache_ttl.get(category, self.cache_ttl['default'])
        
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
        """Load Excel file with LRU caching (from OptimizedDataLoader)"""
        return pd.read_excel(file_path, engine='openpyxl')
    
    @lru_cache(maxsize=128) 
    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with LRU caching (from OptimizedDataLoader)"""
        return pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    
    
    def _standardize_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Standardize column names using ColumnStandardizer"""
        if df is None or df.empty:
            return df
        
        # Use ColumnStandardizer if available
        if ColumnStandardizer:
            df = ColumnStandardizer.standardize_dataframe(df, file_type)
        else:
            # Fallback to basic mappings if ColumnStandardizer not available
            column_mappings = {
                'Planning_Ballance': 'Planning Balance',  # Fix typo
                'Planning_Balance': 'Planning Balance',
                'Theoratical_Balance': 'Theoretical Balance',
                'Theoretical_Balance': 'Theoretical Balance',
                'Yarn_ID': 'Desc#',
                'YarnID': 'Desc#',
                'Yarn ID': 'Desc#',
                'Yarn': 'Desc#',  # For Yarn_Demand files
                'Desc': 'Desc#',
                'Style #': 'Style#',
                'fStyle': 'fStyle#',
                'On_Order': 'On Order',
                'BOM_Percentage': 'BOM_Percent'
            }
            
            # Apply mappings
            df = df.rename(columns=column_mappings)
        
        # Additional file-specific fixes
        if file_type == 'yarn_inventory':
            # Ensure Planning Balance column exists and is calculated correctly
            planning_col = ColumnStandardizer.find_column(df, ['Planning Balance', 'Planning_Balance', 'Planning_Ballance']) if ColumnStandardizer else 'Planning Balance'
            
            if planning_col not in df.columns:
                theoretical_col = ColumnStandardizer.find_column(df, ['Theoretical Balance', 'Theoretical_Balance']) if ColumnStandardizer else 'Theoretical Balance'
                allocated_col = ColumnStandardizer.find_column(df, ['Allocated', 'allocated']) if ColumnStandardizer else 'Allocated'
                on_order_col = ColumnStandardizer.find_column(df, ['On Order', 'On_Order']) if ColumnStandardizer else 'On Order'
                
                if all(col in df.columns for col in [theoretical_col, allocated_col, on_order_col]):
                    # CRITICAL: Allocated is ALREADY NEGATIVE in files
                    df['Planning Balance'] = df[theoretical_col] + df[allocated_col] + df[on_order_col]
                    logger.info("Calculated Planning Balance from components")
        
        return df
    
    def _find_files(self, patterns: List[str]) -> List[Path]:
        """Find files matching patterns across all data paths (enhanced from all loaders)"""
        all_files = []
        
        for data_path in self.data_paths:
            if not data_path.exists():
                continue
                
            for pattern in patterns:
                # Use glob for pattern matching
                files = list(data_path.glob(pattern))
                all_files.extend(files)
                
                # Also search in common subdirectories
                for subdir in ['5', '08-09', '08-06', '08-04', 'archive/08-04', 'ERP Data']:
                    subpath = data_path / subdir
                    if subpath.exists():
                        files = list(subpath.glob(pattern))
                        all_files.extend(files)
        
        # Remove duplicates and sort by modification time (most recent first)
        unique_files = list(set(all_files))
        if unique_files:
            unique_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        return unique_files
    
    def _load_file(self, file_patterns: Union[str, List[str]], file_type: str) -> Optional[pd.DataFrame]:
        """Load a single file with enhanced error handling and multi-path search"""
        try:
            # Convert single pattern to list
            if isinstance(file_patterns, str):
                file_patterns = [file_patterns]
            
            # Find matching files across all paths
            files = self._find_files(file_patterns)
            
            if not files:
                logger.warning(f"No files found for patterns: {file_patterns}")
                return None
            
            # Use most recent file
            file_path = files[0]
            logger.info(f"Loading {file_type} from: {file_path}")
            
            # Load using cached methods for better performance
            if file_path.suffix.lower() == '.xlsx':
                df = self.load_excel_file(str(file_path))
            elif file_path.suffix.lower() == '.csv':
                df = self.load_csv_file(str(file_path))
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Check for header rows that appear as data (common issue)
            if df is not None and len(df) > 0:
                # Check if first row contains column names as values
                first_row_values = df.iloc[0].astype(str).tolist()
                column_names = df.columns.astype(str).tolist()
                
                # If first row looks like headers, drop it
                if any(val in column_names for val in first_row_values if val != 'nan'):
                    logger.warning(f"Detected header row as data in {file_type}, removing...")
                    df = df.iloc[1:].reset_index(drop=True)
                    
                    # Try to convert numeric columns back to proper types
                    for col in df.columns:
                        try:
                            # Skip obvious string columns
                            if 'description' in col.lower() or 'name' in col.lower() or 'color' in col.lower():
                                continue
                            # Try to convert to numeric
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
            
            # Standardize columns
            df = self._standardize_columns(df, file_type)
            
            logger.debug(f"Loaded {file_type}: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_type}: {e}")
            return None
    
    def load_yarn_inventory(self) -> Optional[pd.DataFrame]:
        """Load yarn inventory with database-first, enhanced caching"""
        start_time = time.time()
        
        # Check cache first (enhanced caching from OptimizedDataLoader)
        cached_data = self.load_from_cache('yarn_inventory')
        if cached_data is not None:
            self.load_times['yarn_inventory'] = time.time() - start_time
            return cached_data
        
        df = None
        
        # Try database first (from DatabaseDataLoader)
        if self.use_database and DATABASE_AVAILABLE:
            try:
                self._connect_database()
                if self.db_conn:
                    query = """
                        SELECT 
                            y.desc_id as "Desc#",
                            y.yarn_description as "Yarn Description",
                            yi.theoretical_balance as "Theoretical_Balance",
                            yi.allocated as "Allocated",
                            yi.on_order as "On_Order",
                            yi.planning_balance as "Planning_Balance",
                            yi.weeks_of_supply as "Weeks of Supply",
                            yi.cost_per_pound as "Cost/Pound"
                        FROM production.yarns y
                        LEFT JOIN production.yarn_inventory_ts yi ON y.yarn_id = yi.yarn_id
                        WHERE yi.snapshot_date = (SELECT MAX(snapshot_date) FROM production.yarn_inventory_ts)
                        ORDER BY y.desc_id
                    """
                    df = pd.read_sql_query(query, self.db_conn)
                    logger.info(f"Loaded yarn inventory from database: {len(df)} records")
            except Exception as e:
                logger.warning(f"Database query failed: {e}. Falling back to files.")
        
        # Fallback to file loading with enhanced patterns
        if df is None:
            patterns = [
                'yarn_inventory*.xlsx',
                'Yarn_Inventory*.xlsx', 
                'yarn_inventory*.csv',
                'Yarn_ID*.csv'
            ]
            df = self._load_file(patterns, 'yarn_inventory')
        
        if df is not None:
            # Critical data validation
            required_cols = ['Desc#']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            
            # Save to cache (enhanced caching)
            self.save_to_cache(df, 'yarn_inventory')
            self.cache_stats['loads'] += 1
        
        self.load_times['yarn_inventory'] = time.time() - start_time
        logger.info(f"Loaded yarn inventory: {len(df) if df is not None else 0} records in {self.load_times['yarn_inventory']:.2f}s")
        
        return df
    
    def load_sales_orders(self) -> Optional[pd.DataFrame]:
        """Load sales orders with database-first, enhanced patterns"""
        start_time = time.time()
        
        # Check cache (enhanced)
        cached_data = self.load_from_cache('sales_orders')
        if cached_data is not None:
            self.load_times['sales_orders'] = time.time() - start_time
            return cached_data
        
        df = None
        
        # Try database first
        if self.use_database and DATABASE_AVAILABLE:
            try:
                self._connect_database()
                if self.db_conn:
                    query = """
                        SELECT 
                            so.so_number as "SO#",
                            so.customer_id as "Customer ID",
                            c.customer_name as "Sold To",
                            so.style_number as "fStyle#",
                            so.quantity_ordered as "Ordered",
                            so.quantity_shipped as "Picked/Shipped",
                            so.balance as "Balance",
                            so.ship_date as "Ship Date",
                            so.available_qty as "Available"
                        FROM production.sales_orders_ts so
                        LEFT JOIN production.customers c ON so.customer_id = c.customer_id
                        WHERE so.snapshot_date = (SELECT MAX(snapshot_date) FROM production.sales_orders_ts)
                        ORDER BY so.ship_date
                    """
                    df = pd.read_sql_query(query, self.db_conn)
                    logger.info(f"Loaded sales orders from database: {len(df)} records")
            except Exception as e:
                logger.warning(f"Database query failed: {e}. Falling back to files.")
        
        # Enhanced file patterns from all loaders
        if df is None:
            patterns = [
                'eFab_SO_List*.csv',
                'Sales Activity Report*.csv',
                'Sales Activity*.csv',
                'sales_orders*.xlsx',
                'sales_orders*.csv'
            ]
            df = self._load_file(patterns, 'sales_orders')
        
        if df is not None:
            self.save_to_cache(df, 'sales_orders')
            self.cache_stats['loads'] += 1
        
        self.load_times['sales_orders'] = time.time() - start_time
        logger.info(f"Loaded sales orders: {len(df) if df is not None else 0} records in {self.load_times['sales_orders']:.2f}s")
        
        return df
    
    def load_bom(self) -> Optional[pd.DataFrame]:
        """Load Bill of Materials with database-first, prioritized patterns"""
        start_time = time.time()
        
        # Check cache (enhanced)
        cached_data = self.load_from_cache('bom_data')  # Use consistent key
        if cached_data is not None:
            self.load_times['bom'] = time.time() - start_time
            return cached_data
        
        df = None
        
        # Try database first
        if self.use_database and DATABASE_AVAILABLE:
            try:
                self._connect_database()
                if self.db_conn:
                    query = """
                        SELECT 
                            s.style_number as "Style#",
                            y.desc_id as "desc#",
                            y.yarn_description as "Yarn Description",
                            sb.percentage as "BOM_Percentage",
                            sb.unit as "unit"
                        FROM production.style_bom sb
                        JOIN production.styles s ON sb.style_id = s.style_id
                        JOIN production.yarns y ON sb.yarn_id = y.yarn_id
                        ORDER BY s.style_number, y.desc_id
                    """
                    df = pd.read_sql_query(query, self.db_conn)
                    logger.info(f"Loaded BOM from database: {len(df)} records")
            except Exception as e:
                logger.warning(f"Database query failed: {e}. Falling back to files.")
        
        # File patterns prioritized from ParallelDataLoader (BOM_updated first)
        if df is None:
            patterns = [
                'BOM_updated.csv',      # Highest priority
                'Style_BOM*.csv',
                'BOM_updated*.csv', 
                'BOM_Master*.csv',
                'BOM*.xlsx',
                'BOM*.csv'
            ]
            df = self._load_file(patterns, 'bom')
        
        if df is not None:
            self.save_to_cache(df, 'bom_data')
            self.cache_stats['loads'] += 1
        
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
        logger.info(f"Cache performance: {self.cache_stats['hits']} hits, {self.cache_stats['misses']} misses")
        
        return results
    
    def load_knit_orders(self) -> Optional[pd.DataFrame]:
        """Load knit orders with database-first, enhanced patterns"""
        start_time = time.time()
        
        # Check cache
        cached_data = self.load_from_cache('knit_orders')
        if cached_data is not None:
            self.load_times['knit_orders'] = time.time() - start_time
            return cached_data
        
        df = None
        
        # Try database first
        if self.use_database and DATABASE_AVAILABLE:
            try:
                self._connect_database()
                if self.db_conn:
                    query = """
                        SELECT 
                            ko.ko_number as "KO#",
                            ko.style_number as "Style#",
                            ko.qty_ordered_lbs as "Qty Ordered (lbs)",
                            ko.g00_lbs as "G00 (lbs)",
                            ko.shipped_lbs as "Shipped (lbs)",
                            ko.balance_lbs as "Balance (lbs)",
                            ko.machine as "Machine",
                            ko.start_date as "Start Date",
                            ko.quoted_date as "Quoted Date"
                        FROM production.knit_orders_ts ko
                        WHERE ko.snapshot_date = (SELECT MAX(snapshot_date) FROM production.knit_orders_ts)
                        ORDER BY ko.start_date
                    """
                    df = pd.read_sql_query(query, self.db_conn)
                    logger.info(f"Loaded knit orders from database: {len(df)} records")
            except Exception as e:
                logger.warning(f"Database query failed: {e}. Falling back to files.")
        
        # Enhanced file patterns
        if df is None:
            patterns = [
                'eFab_Knit_Orders*.xlsx',
                'knit_orders*.xlsx',
                'knit_orders*.csv',
                'eFab_Knit_Orders_*.xlsx'
            ]
            df = self._load_file(patterns, 'knit_orders')
        
        if df is not None:
            self.save_to_cache(df, 'knit_orders')
            self.cache_stats['loads'] += 1
        
        self.load_times['knit_orders'] = time.time() - start_time
        logger.info(f"Loaded knit orders: {len(df) if df is not None else 0} records in {self.load_times['knit_orders']:.2f}s")
        
        return df
    
    def load_styles(self) -> Optional[pd.DataFrame]:
        """Load styles master data with enhanced patterns"""
        start_time = time.time()
        
        cached_data = self.load_from_cache('styles')
        if cached_data is not None:
            self.load_times['styles'] = time.time() - start_time
            return cached_data
        
        patterns = [
            'eFab_Styles*.xlsx',
            'styles*.xlsx',
            'styles*.csv',
            'style_master*.xlsx',
            'style_master*.csv'
        ]
        df = self._load_file(patterns, 'styles')
        
        if df is not None:
            self.save_to_cache(df, 'styles')
            self.cache_stats['loads'] += 1
        
        self.load_times['styles'] = time.time() - start_time
        logger.info(f"Loaded styles: {len(df) if df is not None else 0} records in {self.load_times['styles']:.2f}s")
        
        return df
    
    def batch_load_inventory(self) -> Dict[str, pd.DataFrame]:
        """Load all inventory files in parallel (from OptimizedDataLoader)"""
        # Check cache first
        cached_data = self.load_from_cache('all_inventory')
        if cached_data is not None:
            return cached_data
        
        inventory_stages = ['F01', 'G00', 'G02', 'I01']
        inventory_data = {}
        
        def load_stage(stage):
            """Load a single inventory stage with enhanced search"""
            try:
                patterns = [
                    f'eFab_Inventory_{stage}_*.xlsx',
                    f'eFab_Inventory_{stage}_*.csv',
                    f'inventory_{stage.lower()}*.xlsx',
                    f'inventory_{stage.lower()}*.csv'
                ]
                
                files = self._find_files(patterns)
                if files:
                    file_path = files[0]  # Most recent
                    logger.info(f"Loading {stage} from {file_path}")
                    
                    if file_path.suffix.lower() == '.xlsx':
                        data = self.load_excel_file(str(file_path))
                    else:
                        data = self.load_csv_file(str(file_path))
                    
                    return stage, data
                else:
                    logger.warning(f"No files found for stage {stage}")
                    return stage, pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error loading {stage}: {e}")
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
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_files = self.cache_dir.glob("*.pkl")
        count = 0
        for file in cache_files:
            file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")
        self.cache_stats['hits'] = 0
        self.cache_stats['misses'] = 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get loader performance metrics"""
        return {
            'load_times': self.load_times,
            'cache_stats': self.cache_stats,
            'cache_hit_rate': self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0,
            'total_load_time': sum(self.load_times.values()),
            'data_path': str(self.data_path)
        }
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Backward compatibility method - same as load_all_data_parallel
        """
        return self.load_all_data_parallel()
    
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
    

# Backward compatibility aliases for existing imports
OptimizedDataLoader = ConsolidatedDataLoader
ParallelDataLoader = ConsolidatedDataLoader
DatabaseDataLoader = ConsolidatedDataLoader

# Backward compatibility integration functions
def integrate_with_erp(erp_instance):
    """
    Backward compatibility function for optimized_data_loader integration
    """
    loader = ConsolidatedDataLoader()
    
    # Replace load methods with consolidated versions
    original_load = getattr(erp_instance, 'load_all_data', None)
    
    def consolidated_load_all():
        """Replacement for load_all_data using consolidated loader"""
        logger.info("Using consolidated data loading...")
        
        # Load all data using parallel method
        all_data = loader.load_all_data_parallel()
        
        # Assign to ERP instance attributes (maintain compatibility)
        erp_instance.raw_materials_data = all_data.get('yarn_inventory', pd.DataFrame())
        erp_instance.yarn_data = erp_instance.raw_materials_data
        erp_instance.sales_data = all_data.get('sales_orders', pd.DataFrame())
        erp_instance.bom_data = all_data.get('bom', pd.DataFrame())
        
        # Handle knit orders if available
        if hasattr(erp_instance, 'knit_orders'):
            erp_instance.knit_orders = all_data.get('knit_orders', pd.DataFrame())
        
        logger.info(f"Consolidated loading complete: {len(all_data)} data sources")
        
    # Replace the method
    erp_instance.load_all_data = consolidated_load_all
    
    # Add performance methods
    erp_instance.clear_data_cache = loader.clear_cache
    erp_instance.get_performance_metrics = loader.get_performance_metrics
    
    return loader


def quick_parallel_load(data_path: str = None) -> Dict[str, Any]:
    """Backward compatibility function from parallel_data_loader"""
    loader = ConsolidatedDataLoader(data_path)
    return loader.load_all_data_parallel()


def integrate_database_loader(erp_instance):
    """Backward compatibility function from database_data_loader"""
    loader = ConsolidatedDataLoader()
    
    # Override load methods with database-first approach
    original_load = getattr(erp_instance, 'load_data', None)
    
    def enhanced_load():
        """Enhanced data loading with database support"""
        try:
            # Use consolidated loader with database support
            erp_instance.yarn_inventory = loader.load_yarn_inventory()
            erp_instance.sales_orders = loader.load_sales_orders()
            erp_instance.knit_orders = loader.load_knit_orders()
            erp_instance.bom = loader.load_bom()
            
            logger.info("Data loaded from consolidated loader successfully")
            return True
        except Exception as e:
            logger.warning(f"Consolidated loading failed: {e}. Falling back to original.")
            return original_load() if original_load else False
    
    erp_instance.load_data = enhanced_load
    erp_instance.database_loader = loader
    
    return erp_instance


# Legacy class alias for maximum compatibility
class UnifiedDataLoader(ConsolidatedDataLoader):
    """Legacy alias for ConsolidatedDataLoader"""
    pass


if __name__ == "__main__":
    test_unified_loader()