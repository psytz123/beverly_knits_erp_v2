#!/usr/bin/env python3
"""
Day 0 Emergency Fixes for Beverly Knits ERP System
Production-ready fixes with dynamic path resolution, column handling, price parsing, and netting logic

This script provides comprehensive emergency fixes that can be imported into the main ERP system:
1. Dynamic path resolution with multiple fallback paths
2. Robust column alias system for handling variations
3. Advanced price string parsing for complex formats  
4. Real KPI calculations using actual data
5. Complete multi-level BOM netting logic

Author: Beverly Knits ERP System
Version: 1.0.0
Date: 2025-09-02
"""

import os
import sys
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('day0_emergency_fixes')

class DynamicPathResolver:
    """
    Advanced path resolution system with multiple fallback strategies
    Handles all data file variations and locations with comprehensive error handling
    """
    
    def __init__(self):
        """Initialize path resolver with comprehensive fallback paths"""
        self.logger = logging.getLogger(f'{__name__}.DynamicPathResolver')
        
        # Primary data paths in order of preference
        self.primary_paths = [
            "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data",
            "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data",
            "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5",
            "/mnt/c/Users/psytz/sc data/ERP Data"
        ]
        
        # Dated subdirectories to check (newest first)
        self.dated_subdirs = [
            "8-28-2025", "8-26-2025", "8-24-2025", "8-22-2025",
            "2025-08-28", "2025-08-26", "2025-08-24", "2025-08-22"
        ]
        
        # Mock/test data fallback paths
        self.fallback_paths = [
            "/mnt/c/finalee/beverly_knits_erp_v2/data/mock/5/ERP Data",
            "/mnt/c/finalee/beverly_knits_erp_v2/data/mock/5",
            "/mnt/c/finalee/beverly_knits_erp_v2/data/sample"
        ]
        
        # File name patterns for different data types
        self.file_patterns = {
            'yarn_inventory': [
                'yarn_inventory.xlsx',
                'yarn_inventory (*.xlsx',
                'Yarn_Inventory*.xlsx',
                'yarn_inventory*.csv'
            ],
            'knit_orders': [
                'eFab_Knit_Orders.xlsx',
                'eFab_Knit_Orders_*.xlsx',
                'Knit_Orders*.xlsx',
                'knit_orders*.xlsx'
            ],
            'bom': [
                'BOM_updated.csv',
                'Style_BOM.csv',
                'BOM*.csv',
                'bom*.csv'
            ],
            'sales_orders': [
                'eFab_SO_List.xlsx',
                'eFab_SO_List_*.xlsx',
                'Sales_Orders*.xlsx',
                'sales_orders*.xlsx'
            ],
            'sales_activity': [
                'Sales Activity Report.csv',
                'Sales_Activity_Report*.csv',
                'sales_activity*.csv'
            ],
            'yarn_demand': [
                'Yarn_Demand.xlsx',
                'Yarn_Demand_*.xlsx',
                'yarn_demand*.xlsx'
            ],
            'yarn_demand_by_style': [
                'Yarn_Demand_By_Style.xlsx',
                'Yarn_Demand_By_Style_*.xlsx',
                'yarn_demand_by_style*.xlsx'
            ],
            'inventory_f01': [
                'eFab_Inventory_F01.xlsx',
                'eFab_Inventory_F01_*.xlsx',
                'inventory_f01*.xlsx'
            ],
            'inventory_g00': [
                'eFab_Inventory_G00.xlsx',
                'eFab_Inventory_G00_*.xlsx',
                'inventory_g00*.xlsx'
            ],
            'inventory_g02': [
                'eFab_Inventory_G02.xlsx',
                'eFab_Inventory_G02_*.xlsx',
                'inventory_g02*.xlsx'
            ],
            'inventory_i01': [
                'eFab_Inventory_I01.xlsx',
                'eFab_Inventory_I01_*.xlsx',
                'inventory_i01*.xlsx'
            ],
            'yarn_id': [
                'Yarn_ID_Master.csv',
                'Yarn_ID.csv',
                'yarn_id*.csv'
            ],
            'supplier': [
                'Supplier_ID.csv',
                'supplier*.csv'
            ]
        }
        
        # Cache for resolved paths
        self._path_cache = {}
        
    def resolve_data_file(self, file_type: str, use_cache: bool = True) -> Optional[str]:
        """
        Resolve the full path to a data file with comprehensive fallback logic
        
        Args:
            file_type: Type of file to find (e.g., 'yarn_inventory', 'bom')
            use_cache: Whether to use cached paths for performance
            
        Returns:
            Full path to the data file or None if not found
        """
        cache_key = f"{file_type}"
        
        if use_cache and cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            if os.path.exists(cached_path):
                return cached_path
            else:
                # Remove invalid cached path
                del self._path_cache[cache_key]
        
        if file_type not in self.file_patterns:
            self.logger.warning(f"Unknown file type: {file_type}")
            return None
        
        patterns = self.file_patterns[file_type]
        
        # Search in all path combinations
        for base_path in self.primary_paths + self.fallback_paths:
            if not os.path.exists(base_path):
                continue
                
            # Check base path first
            file_path = self._find_file_in_directory(base_path, patterns)
            if file_path:
                self._path_cache[cache_key] = file_path
                self.logger.info(f"Found {file_type} at: {file_path}")
                return file_path
            
            # Check dated subdirectories
            for subdir in self.dated_subdirs:
                dated_path = os.path.join(base_path, subdir)
                if os.path.exists(dated_path):
                    file_path = self._find_file_in_directory(dated_path, patterns)
                    if file_path:
                        self._path_cache[cache_key] = file_path
                        self.logger.info(f"Found {file_type} at: {file_path}")
                        return file_path
        
        self.logger.error(f"Could not find {file_type} file in any location")
        return None
    
    def _find_file_in_directory(self, directory: str, patterns: List[str]) -> Optional[str]:
        """Find first matching file in directory using patterns"""
        try:
            for pattern in patterns:
                if '*' in pattern:
                    # Use glob for wildcard patterns
                    import glob
                    matches = glob.glob(os.path.join(directory, pattern))
                    if matches:
                        # Sort by modification time, newest first
                        matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        return matches[0]
                else:
                    # Direct file check
                    file_path = os.path.join(directory, pattern)
                    if os.path.exists(file_path):
                        return file_path
        except Exception as e:
            self.logger.debug(f"Error searching in {directory}: {e}")
            
        return None
    
    def resolve_all_files(self) -> Dict[str, str]:
        """
        Resolve paths for all known file types
        
        Returns:
            Dictionary mapping file types to resolved paths
        """
        resolved_files = {}
        
        for file_type in self.file_patterns.keys():
            path = self.resolve_data_file(file_type)
            if path:
                resolved_files[file_type] = path
            else:
                self.logger.warning(f"Could not resolve path for {file_type}")
                
        self.logger.info(f"Successfully resolved {len(resolved_files)}/{len(self.file_patterns)} file paths")
        return resolved_files
    
    def validate_data_integrity(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """
        Validate data file integrity and return metadata
        
        Args:
            file_paths: Dictionary of file_type -> path mappings
            
        Returns:
            Dictionary of validation results and metadata
        """
        validation_results = {}
        
        for file_type, path in file_paths.items():
            try:
                stat_info = os.stat(path)
                
                # Basic file info
                file_info = {
                    'path': path,
                    'exists': True,
                    'size_bytes': stat_info.st_size,
                    'modified': datetime.fromtimestamp(stat_info.st_mtime),
                    'readable': os.access(path, os.R_OK)
                }
                
                # Try to read file to check format
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path, nrows=1)
                    else:
                        df = pd.read_excel(path, nrows=1)
                    
                    file_info.update({
                        'format_valid': True,
                        'columns_count': len(df.columns),
                        'sample_columns': list(df.columns)[:5]
                    })
                    
                except Exception as e:
                    file_info.update({
                        'format_valid': False,
                        'format_error': str(e)
                    })
                
                validation_results[file_type] = file_info
                
            except Exception as e:
                validation_results[file_type] = {
                    'path': path,
                    'exists': False,
                    'error': str(e)
                }
        
        return validation_results


class ColumnAliasSystem:
    """
    Advanced column alias system handling all known variations
    Provides robust column detection and standardization
    """
    
    def __init__(self):
        """Initialize with comprehensive column mappings"""
        self.logger = logging.getLogger(f'{__name__}.ColumnAliasSystem')
        
        # Comprehensive column alias mappings
        self.aliases = {
            # Yarn identifiers with all variations
            'yarn_id': [
                'Desc#', 'desc#', 'Desc #', 'desc #',
                'Yarn', 'yarn', 'YarnID', 'Yarn_ID', 'yarn_id',
                'Material_ID', 'MaterialID', 'material_id',
                'Desc', 'Description_ID', 'Item_ID'
            ],
            
            # Style identifiers - careful distinction
            'style_id': [
                'Style#', 'Style #', 'style', 'Style', 'style_id', 'Style_ID'
            ],
            'fstyle_id': [
                'fStyle#', 'fStyle', 'f_style', 'FStyle', 'FStyle#'
            ],
            'gstyle_id': [
                'gStyle', 'gStyle#', 'g_style', 'GStyle', 'GStyle#'
            ],
            
            # Planning Balance - critical column with typos
            'planning_balance': [
                'Planning Balance', 'Planning_Balance', 
                'Planning_Ballance', 'planning_balance',  # Handle typo
                'Available_Balance', 'Available Balance',
                'Current_Balance', 'Balance'
            ],
            
            # Other balance types
            'beginning_balance': [
                'Beginning Balance', 'Beginning_Balance', 'beginning_balance',
                'Starting_Balance', 'Start_Balance', 'Initial_Balance'
            ],
            'theoretical_balance': [
                'Theoretical Balance', 'Theoretical_Balance', 'theoretical_balance',
                'Calculated_Balance', 'Expected_Balance'
            ],
            
            # Quantities with units
            'qty_ordered_lbs': [
                'Qty Ordered (lbs)', 'qty_ordered_lbs', 'Ordered (lbs)',
                'Quantity_Ordered_Lbs', 'Order_Qty_Lbs'
            ],
            'balance_lbs': [
                'Balance (lbs)', 'balance_lbs', 'Balance_lbs',
                'Remaining_Lbs', 'Outstanding_Lbs'
            ],
            'shipped_lbs': [
                'Shipped (lbs)', 'shipped_lbs', 'Shipped_lbs',
                'Delivered_Lbs', 'Sent_Lbs'
            ],
            
            # Production stages
            'g00_lbs': [
                'G00 (lbs)', 'g00_lbs', 'Stage_G00', 'G00_Stage',
                'Greige_Lbs', 'Raw_Lbs'
            ],
            'g02_lbs': [
                'G02 (lbs)', 'g02_lbs', 'Stage_G02', 'G02_Stage'
            ],
            'i01_lbs': [
                'I01 (lbs)', 'i01_lbs', 'Stage_I01', 'I01_Stage',
                'Inspection_Lbs', 'QC_Lbs'
            ],
            'f01_lbs': [
                'F01 (lbs)', 'f01_lbs', 'Stage_F01', 'F01_Stage',
                'Finished_Lbs', 'Final_Lbs'
            ],
            
            # Order numbers
            'order_number': [
                'Order #', 'Order#', 'order_number', 'Order_Number',
                'OrderNo', 'Order_No'
            ],
            'po_number': [
                'PO#', 'PO #', 'po_number', 'PO_Number',
                'Purchase_Order', 'PurchaseOrder'
            ],
            'so_number': [
                'SO #', 'SO#', 'so_number', 'Sales_Order',
                'SalesOrder', 'SO_Number'
            ],
            'ko_number': [
                'KO #', 'KO#', 'Actions', 'Knit_Order',
                'KnitOrder', 'KO_Number'
            ],
            
            # Dates
            'start_date': [
                'Start Date', 'start_date', 'Begin_Date',
                'StartDate', 'Commencement_Date'
            ],
            'ship_date': [
                'Ship Date', 'ship_date', 'Shipping_Date',
                'ShipDate', 'Delivery_Date'
            ],
            'quoted_date': [
                'Quoted Date', 'quoted_date', 'Quote_Date',
                'QuoteDate', 'Estimation_Date'
            ],
            
            # BOM specific
            'bom_percentage': [
                'BOM_Percent', 'BOM_Percentage', 'Percentage',
                'BOM%', 'bom_percentage', 'Usage_Percentage',
                'Material_Percentage', 'Composition_Percent'
            ],
            
            # Cost and price fields
            'cost_per_pound': [
                'Cost/Pound', 'Cost_Pound', 'cost_per_pound',
                'Unit_Cost', 'Price_Per_Pound', 'Cost_Per_Lb'
            ],
            'unit_price': [
                'Unit Price', 'unit_price', 'Price', 'Unit_Price',
                'Item_Price', 'Single_Price'
            ],
            'total_cost': [
                'Total Cost', 'Total_Cost', 'total_cost',
                'Total_Cast', 'Total_Value',  # Handle typo
                'Extended_Cost', 'Line_Cost'
            ],
            'line_price': [
                'Line Price', 'line_price', 'LinePrice',
                'Extended_Price', 'Total_Price'
            ],
            
            # Inventory movements
            'on_order': [
                'On Order', 'On_Order', 'on_order',
                'Ordered', 'Qty_On_Order', 'Outstanding_Orders'
            ],
            'allocated': [
                'Allocated', 'allocated', 'Reserved',
                'Qty_Allocated', 'Allocation', 'Committed'
            ],
            'received': [
                'Received', 'received', 'Receipts',
                'Qty_Received', 'Incoming', 'Delivered'
            ],
            'consumed': [
                'Consumed', 'consumed', 'Usage',
                'Qty_Used', 'Used', 'Issued'
            ],
            
            # Supplier and customer
            'supplier': [
                'Supplier', 'supplier', 'Vendor', 'vendor',
                'Supplier_Name', 'Purchased From', 'Source'
            ],
            'customer': [
                'Customer', 'customer', 'Client', 'client',
                'Customer_Name', 'Sold To', 'Ship_To'
            ],
            
            # Description
            'description': [
                'Description', 'Desc', 'description', 'desc',
                'Material_Name', 'Item_Description', 'Product_Name'
            ],
            
            # Status
            'status': [
                'Status', 'status', 'Order_Status', 'State',
                'Current_Status', 'Progress_Status'
            ],
            
            # Units of measure
            'unit': [
                'UOM', 'uom', 'Unit', 'unit',
                'Unit_Of_Measure', 'Measure_Unit'
            ],
            
            # Quality indicators
            'good_qty': [
                'Good Ea.', 'good_qty', 'Good_Quantity',
                'Acceptable', 'Pass_Qty'
            ],
            'bad_qty': [
                'Bad Ea.', 'bad_qty', 'Defect_Quantity',
                'Rejected', 'Fail_Qty', 'Seconds'
            ]
        }
        
        # Create reverse mapping for quick lookup
        self._reverse_mapping = {}
        for standard_name, variations in self.aliases.items():
            for variation in variations:
                self._reverse_mapping[variation] = standard_name
    
    def find_column(self, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """
        Find column in DataFrame by type using alias system
        
        Args:
            df: DataFrame to search
            column_type: Standard column type (e.g., 'yarn_id', 'planning_balance')
            
        Returns:
            Actual column name found in DataFrame or None
        """
        if column_type not in self.aliases:
            self.logger.warning(f"Unknown column type: {column_type}")
            return None
        
        for alias in self.aliases[column_type]:
            if alias in df.columns:
                self.logger.debug(f"Found {column_type} as '{alias}'")
                return alias
                
        self.logger.warning(f"Column type '{column_type}' not found in DataFrame")
        return None
    
    def get_value(self, row: pd.Series, column_type: str, default=None) -> Any:
        """
        Get value from row using column alias system
        
        Args:
            row: DataFrame row (Series)
            column_type: Standard column type
            default: Default value if column not found
            
        Returns:
            Value from row or default
        """
        if column_type not in self.aliases:
            return default
            
        for alias in self.aliases[column_type]:
            if alias in row and pd.notna(row[alias]):
                return row[alias]
                
        return default
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename DataFrame columns to standard names
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized column names
        """
        df_copy = df.copy()
        rename_mapping = {}
        
        for col in df_copy.columns:
            if col in self._reverse_mapping:
                standard_name = self._reverse_mapping[col]
                rename_mapping[col] = standard_name
        
        if rename_mapping:
            df_copy = df_copy.rename(columns=rename_mapping)
            self.logger.info(f"Standardized {len(rename_mapping)} columns")
        
        return df_copy
    
    def validate_required_columns(self, df: pd.DataFrame, required_types: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that required column types exist in DataFrame
        
        Args:
            df: DataFrame to validate
            required_types: List of required column types
            
        Returns:
            Tuple of (is_valid, missing_types)
        """
        missing_types = []
        
        for column_type in required_types:
            if not self.find_column(df, column_type):
                missing_types.append(column_type)
        
        is_valid = len(missing_types) == 0
        return is_valid, missing_types


class PriceStringParser:
    """
    Advanced price string parser handling complex formats
    Supports currency symbols, units, formatting, and edge cases
    """
    
    def __init__(self):
        """Initialize parser with regex patterns"""
        self.logger = logging.getLogger(f'{__name__}.PriceStringParser')
        
        # Compiled regex patterns for performance
        self.patterns = {
            # Standard currency with optional units: $4.07, $14.95 (kg), $1,234.56
            'currency_with_units': re.compile(
                r'^\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:\(([^)]+)\))?',
                re.IGNORECASE
            ),
            
            # Price with currency symbols: $4.07, USD 14.95, £100.50
            'currency_symbol': re.compile(
                r'^([£$€¥]|USD|EUR|GBP|CAD|AUD)?\s*([0-9,]+(?:\.[0-9]+)?)',
                re.IGNORECASE
            ),
            
            # Price per unit: 4.07/lb, 14.95 per kg, 100.50 each
            'price_per_unit': re.compile(
                r'^([0-9,]+(?:\.[0-9]+)?)\s*(?:per|/)\s*(.+)$',
                re.IGNORECASE
            ),
            
            # Range prices: $10.00-$15.00, 4.07 to 8.15
            'price_range': re.compile(
                r'^[£$€¥]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:-|to)\s*[£$€¥]?\s*([0-9,]+(?:\.[0-9]+)?)',
                re.IGNORECASE
            ),
            
            # Scientific notation: 1.5E+3, 2.5e-2
            'scientific': re.compile(
                r'^([0-9,]+(?:\.[0-9]+)?)[eE]([+-]?[0-9]+)$',
                re.IGNORECASE
            ),
            
            # Percentage: 15%, 0.15
            'percentage': re.compile(
                r'^([0-9,]+(?:\.[0-9]+)?)\s*%?$'
            ),
            
            # Plain number with thousand separators: 1,234.56, 1 234,56
            'plain_number': re.compile(
                r'^([0-9, ]+(?:[.,][0-9]+)?)$'
            )
        }
    
    def parse_price(self, price_str: Union[str, float, int, None], 
                   default_currency: str = 'USD') -> Dict[str, Any]:
        """
        Parse price string and extract numeric value and metadata
        
        Args:
            price_str: Price string to parse
            default_currency: Default currency if none specified
            
        Returns:
            Dictionary with parsed price information:
            {
                'value': float,           # Numeric value
                'currency': str,          # Currency code
                'unit': str,             # Unit of measure
                'raw': str,              # Original string
                'is_valid': bool,        # Whether parsing succeeded
                'is_range': bool,        # Whether this is a price range
                'range_min': float,      # If range, minimum value
                'range_max': float,      # If range, maximum value
                'error': str             # Error message if parsing failed
            }
        """
        result = {
            'value': 0.0,
            'currency': default_currency,
            'unit': None,
            'raw': str(price_str),
            'is_valid': False,
            'is_range': False,
            'range_min': None,
            'range_max': None,
            'error': None
        }
        
        # Handle None, NaN, and empty values
        if price_str is None or price_str == '' or (isinstance(price_str, float) and np.isnan(price_str)):
            result['error'] = 'Empty or null value'
            return result
        
        # Handle numeric types directly
        if isinstance(price_str, (int, float)):
            try:
                result['value'] = float(price_str)
                result['is_valid'] = True
                return result
            except (ValueError, OverflowError) as e:
                result['error'] = f'Invalid numeric value: {e}'
                return result
        
        # Convert to string and clean
        price_str = str(price_str).strip()
        if not price_str:
            result['error'] = 'Empty string'
            return result
        
        # Try each pattern in order of specificity
        try:
            # 1. Check for price ranges first
            range_match = self.patterns['price_range'].match(price_str)
            if range_match:
                min_val = self._clean_numeric_string(range_match.group(1))
                max_val = self._clean_numeric_string(range_match.group(2))
                
                result.update({
                    'value': (min_val + max_val) / 2,  # Average of range
                    'is_valid': True,
                    'is_range': True,
                    'range_min': min_val,
                    'range_max': max_val
                })
                return result
            
            # 2. Currency with optional units: $4.07, $14.95 (kg)
            currency_match = self.patterns['currency_with_units'].match(price_str)
            if currency_match:
                value_str = currency_match.group(1)
                unit = currency_match.group(2)
                
                numeric_value = self._clean_numeric_string(value_str)
                
                result.update({
                    'value': numeric_value,
                    'unit': unit,
                    'is_valid': True
                })
                
                # Extract currency if present
                if price_str.startswith('$'):
                    result['currency'] = 'USD'
                elif price_str.startswith('£'):
                    result['currency'] = 'GBP'
                elif price_str.startswith('€'):
                    result['currency'] = 'EUR'
                
                return result
            
            # 3. Price per unit: 4.07/lb, 14.95 per kg
            per_unit_match = self.patterns['price_per_unit'].match(price_str)
            if per_unit_match:
                value_str = per_unit_match.group(1)
                unit = per_unit_match.group(2).strip()
                
                numeric_value = self._clean_numeric_string(value_str)
                
                result.update({
                    'value': numeric_value,
                    'unit': unit,
                    'is_valid': True
                })
                return result
            
            # 4. Scientific notation
            sci_match = self.patterns['scientific'].match(price_str)
            if sci_match:
                base = float(sci_match.group(1).replace(',', ''))
                exponent = int(sci_match.group(2))
                
                result.update({
                    'value': base * (10 ** exponent),
                    'is_valid': True
                })
                return result
            
            # 5. Plain number (last resort)
            plain_match = self.patterns['plain_number'].match(price_str)
            if plain_match:
                value_str = plain_match.group(1)
                numeric_value = self._clean_numeric_string(value_str)
                
                result.update({
                    'value': numeric_value,
                    'is_valid': True
                })
                return result
                
        except Exception as e:
            self.logger.error(f"Error parsing price '{price_str}': {e}")
            result['error'] = str(e)
            return result
        
        # If no patterns matched
        result['error'] = f"Could not parse price format: '{price_str}'"
        return result
    
    def _clean_numeric_string(self, value_str: str) -> float:
        """
        Clean and convert numeric string to float
        Handles thousand separators and decimal points
        """
        if not value_str:
            return 0.0
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[£$€¥]', '', value_str).strip()
        
        # Handle different decimal separators
        # European format: 1.234,56 vs US format: 1,234.56
        if ',' in cleaned and '.' in cleaned:
            # Both present - determine which is decimal
            last_comma = cleaned.rfind(',')
            last_dot = cleaned.rfind('.')
            
            if last_comma > last_dot:
                # European format: 1.234,56
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.56
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Only comma - could be thousands or decimal
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely decimal: 1234,56
                cleaned = cleaned.replace(',', '.')
            else:
                # Likely thousands: 1,234,567
                cleaned = cleaned.replace(',', '')
        
        # Remove any remaining non-numeric characters except decimal point
        cleaned = re.sub(r'[^0-9.]', '', cleaned)
        
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            self.logger.warning(f"Could not convert '{cleaned}' to float")
            return 0.0
    
    def parse_batch(self, price_list: List[Union[str, float, int]]) -> List[Dict[str, Any]]:
        """
        Parse a batch of price strings efficiently
        
        Args:
            price_list: List of price strings/numbers to parse
            
        Returns:
            List of parsed price dictionaries
        """
        return [self.parse_price(price) for price in price_list]
    
    def validate_price_range(self, parsed_prices: List[Dict[str, Any]], 
                           min_expected: float = 0.0, 
                           max_expected: float = 10000.0) -> Dict[str, Any]:
        """
        Validate that parsed prices fall within expected ranges
        
        Args:
            parsed_prices: List of parsed price dictionaries
            min_expected: Minimum expected price value
            max_expected: Maximum expected price value
            
        Returns:
            Validation summary
        """
        valid_prices = [p for p in parsed_prices if p['is_valid']]
        invalid_prices = [p for p in parsed_prices if not p['is_valid']]
        
        out_of_range = []
        for p in valid_prices:
            if not (min_expected <= p['value'] <= max_expected):
                out_of_range.append(p)
        
        return {
            'total_count': len(parsed_prices),
            'valid_count': len(valid_prices),
            'invalid_count': len(invalid_prices),
            'out_of_range_count': len(out_of_range),
            'success_rate': len(valid_prices) / len(parsed_prices) if parsed_prices else 0.0,
            'invalid_prices': invalid_prices[:10],  # Sample of invalid prices
            'out_of_range_prices': out_of_range[:10]  # Sample of out-of-range prices
        }


class RealKPICalculator:
    """
    Calculate real KPIs from actual data instead of returning zeros
    Provides comprehensive business metrics with proper data handling
    """
    
    def __init__(self, path_resolver: DynamicPathResolver, column_system: ColumnAliasSystem):
        """Initialize with data access components"""
        self.logger = logging.getLogger(f'{__name__}.RealKPICalculator')
        self.path_resolver = path_resolver
        self.column_system = column_system
        self.price_parser = PriceStringParser()
        
        # Cache for loaded data
        self._data_cache = {}
    
    def calculate_comprehensive_kpis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive KPIs from actual data
        
        Args:
            force_refresh: Whether to force reload of all data
            
        Returns:
            Dictionary of calculated KPIs
        """
        try:
            # Load all necessary data
            data = self._load_all_data(force_refresh)
            
            kpis = {
                'timestamp': datetime.now().isoformat(),
                'data_sources_loaded': len(data),
                'calculation_status': 'success'
            }
            
            # Inventory KPIs
            inventory_kpis = self._calculate_inventory_kpis(data)
            kpis.update(inventory_kpis)
            
            # Production KPIs  
            production_kpis = self._calculate_production_kpis(data)
            kpis.update(production_kpis)
            
            # Sales KPIs
            sales_kpis = self._calculate_sales_kpis(data)
            kpis.update(sales_kpis)
            
            # Financial KPIs
            financial_kpis = self._calculate_financial_kpis(data)
            kpis.update(financial_kpis)
            
            # Planning KPIs
            planning_kpis = self._calculate_planning_kpis(data)
            kpis.update(planning_kpis)
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'calculation_status': 'error',
                'error_message': str(e),
                'data_sources_loaded': 0
            }
    
    def _load_all_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Load all required data files"""
        if not force_refresh and self._data_cache:
            return self._data_cache
        
        data = {}
        
        # Define required data files
        required_files = [
            'yarn_inventory', 'knit_orders', 'bom', 'sales_orders',
            'inventory_f01', 'inventory_g00', 'inventory_g02', 'inventory_i01'
        ]
        
        for file_type in required_files:
            file_path = self.path_resolver.resolve_data_file(file_type)
            if file_path:
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    # Apply column standardization
                    df = self.column_system.standardize_columns(df)
                    data[file_type] = df
                    
                    self.logger.info(f"Loaded {file_type}: {len(df)} rows")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {file_type} from {file_path}: {e}")
            else:
                self.logger.warning(f"Could not find {file_type}")
        
        self._data_cache = data
        return data
    
    def _calculate_inventory_kpis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate inventory-related KPIs"""
        kpis = {}
        
        try:
            if 'yarn_inventory' in data:
                yarn_df = data['yarn_inventory']
                
                # Total yarn items
                kpis['total_yarn_items'] = len(yarn_df)
                
                # Planning balance statistics
                balance_col = self.column_system.find_column(yarn_df, 'planning_balance')
                if balance_col:
                    balances = pd.to_numeric(yarn_df[balance_col], errors='coerce')
                    kpis.update({
                        'total_planning_balance': float(balances.sum()),
                        'average_planning_balance': float(balances.mean()),
                        'negative_balance_items': int((balances < 0).sum()),
                        'zero_balance_items': int((balances == 0).sum()),
                        'positive_balance_items': int((balances > 0).sum())
                    })
                
                # On order quantities
                on_order_col = self.column_system.find_column(yarn_df, 'on_order')
                if on_order_col:
                    on_order = pd.to_numeric(yarn_df[on_order_col], errors='coerce')
                    kpis['total_on_order'] = float(on_order.sum())
            
            # Production stage inventories
            stage_totals = {}
            for stage in ['f01', 'g00', 'g02', 'i01']:
                inventory_key = f'inventory_{stage}'
                if inventory_key in data:
                    stage_df = data[inventory_key]
                    
                    # Find quantity column (varies by stage)
                    qty_col = None
                    for col in stage_df.columns:
                        if 'lbs' in col.lower() and stage.upper() in col:
                            qty_col = col
                            break
                    
                    if qty_col:
                        qty_values = pd.to_numeric(stage_df[qty_col], errors='coerce')
                        stage_totals[f'{stage}_total_lbs'] = float(qty_values.sum())
                    
                    stage_totals[f'{stage}_item_count'] = len(stage_df)
            
            kpis.update(stage_totals)
            
        except Exception as e:
            self.logger.error(f"Error calculating inventory KPIs: {e}")
            kpis['inventory_calculation_error'] = str(e)
        
        return kpis
    
    def _calculate_production_kpis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate production-related KPIs"""
        kpis = {}
        
        try:
            if 'knit_orders' in data:
                knit_df = data['knit_orders']
                
                # Total knit orders
                kpis['total_knit_orders'] = len(knit_df)
                
                # Ordered quantities
                qty_ordered_col = self.column_system.find_column(knit_df, 'qty_ordered_lbs')
                if qty_ordered_col:
                    ordered_qty = pd.to_numeric(knit_df[qty_ordered_col], errors='coerce')
                    kpis['total_ordered_lbs'] = float(ordered_qty.sum())
                    kpis['average_order_size_lbs'] = float(ordered_qty.mean())
                
                # Balance/remaining quantities
                balance_col = self.column_system.find_column(knit_df, 'balance_lbs')
                if balance_col:
                    balance_qty = pd.to_numeric(knit_df[balance_col], errors='coerce')
                    kpis['total_balance_lbs'] = float(balance_qty.sum())
                    
                    # Calculate completion rate
                    if qty_ordered_col:
                        completed_lbs = ordered_qty - balance_qty
                        completion_rate = (completed_lbs.sum() / ordered_qty.sum()) * 100
                        kpis['production_completion_rate_pct'] = float(completion_rate)
                
                # Orders by status
                status_col = self.column_system.find_column(knit_df, 'status')
                if status_col:
                    status_counts = knit_df[status_col].value_counts().to_dict()
                    kpis['orders_by_status'] = status_counts
                
                # Date analysis
                start_date_col = self.column_system.find_column(knit_df, 'start_date')
                if start_date_col:
                    # Orders starting this week/month
                    today = datetime.now()
                    week_start = today - timedelta(days=today.weekday())
                    month_start = today.replace(day=1)
                    
                    try:
                        knit_df[start_date_col] = pd.to_datetime(knit_df[start_date_col], errors='coerce')
                        orders_this_week = (knit_df[start_date_col] >= week_start).sum()
                        orders_this_month = (knit_df[start_date_col] >= month_start).sum()
                        
                        kpis.update({
                            'orders_starting_this_week': int(orders_this_week),
                            'orders_starting_this_month': int(orders_this_month)
                        })
                    except Exception as e:
                        self.logger.debug(f"Date analysis error: {e}")
                
        except Exception as e:
            self.logger.error(f"Error calculating production KPIs: {e}")
            kpis['production_calculation_error'] = str(e)
        
        return kpis
    
    def _calculate_sales_kpis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate sales-related KPIs"""
        kpis = {}
        
        try:
            if 'sales_orders' in data:
                sales_df = data['sales_orders']
                
                # Total sales orders
                kpis['total_sales_orders'] = len(sales_df)
                
                # Customer analysis
                customer_col = self.column_system.find_column(sales_df, 'customer')
                if customer_col:
                    unique_customers = sales_df[customer_col].nunique()
                    kpis['unique_customers'] = int(unique_customers)
                    
                    # Top customers
                    top_customers = sales_df[customer_col].value_counts().head(5).to_dict()
                    kpis['top_customers'] = top_customers
            
            # Additional sales analysis from other files if available
            # This would include sales activity report analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating sales KPIs: {e}")
            kpis['sales_calculation_error'] = str(e)
        
        return kpis
    
    def _calculate_financial_kpis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate financial KPIs"""
        kpis = {}
        
        try:
            total_inventory_value = 0.0
            total_on_order_value = 0.0
            
            if 'yarn_inventory' in data:
                yarn_df = data['yarn_inventory']
                
                # Calculate inventory value
                balance_col = self.column_system.find_column(yarn_df, 'planning_balance')
                cost_col = self.column_system.find_column(yarn_df, 'cost_per_pound')
                
                if balance_col and cost_col:
                    balances = pd.to_numeric(yarn_df[balance_col], errors='coerce').fillna(0)
                    
                    # Parse cost strings
                    costs = []
                    for cost_str in yarn_df[cost_col]:
                        parsed = self.price_parser.parse_price(cost_str)
                        costs.append(parsed['value'] if parsed['is_valid'] else 0.0)
                    
                    costs = pd.Series(costs)
                    inventory_values = balances * costs
                    total_inventory_value = float(inventory_values.sum())
                
                # Calculate on-order value
                on_order_col = self.column_system.find_column(yarn_df, 'on_order')
                if on_order_col and cost_col:
                    on_order_qty = pd.to_numeric(yarn_df[on_order_col], errors='coerce').fillna(0)
                    on_order_values = on_order_qty * costs
                    total_on_order_value = float(on_order_values.sum())
            
            kpis.update({
                'total_inventory_value_usd': total_inventory_value,
                'total_on_order_value_usd': total_on_order_value,
                'total_committed_capital_usd': total_inventory_value + total_on_order_value
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating financial KPIs: {e}")
            kpis['financial_calculation_error'] = str(e)
        
        return kpis
    
    def _calculate_planning_kpis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate planning and forecasting KPIs"""
        kpis = {}
        
        try:
            # Shortage analysis
            if 'yarn_inventory' in data and 'bom' in data:
                yarn_df = data['yarn_inventory']
                bom_df = data['bom']
                
                # Find items with negative planning balance (shortages)
                balance_col = self.column_system.find_column(yarn_df, 'planning_balance')
                if balance_col:
                    balances = pd.to_numeric(yarn_df[balance_col], errors='coerce')
                    
                    shortage_items = yarn_df[balances < 0]
                    kpis.update({
                        'yarn_shortage_count': len(shortage_items),
                        'total_shortage_lbs': float(abs(balances[balances < 0].sum())),
                        'shortage_percentage': float(len(shortage_items) / len(yarn_df) * 100)
                    })
                
                # BOM coverage analysis
                yarn_ids = set(yarn_df[self.column_system.find_column(yarn_df, 'yarn_id')] if 
                              self.column_system.find_column(yarn_df, 'yarn_id') else [])
                bom_yarns = set(bom_df[self.column_system.find_column(bom_df, 'yarn_id')] if 
                               self.column_system.find_column(bom_df, 'yarn_id') else [])
                
                if yarn_ids and bom_yarns:
                    covered_yarns = yarn_ids.intersection(bom_yarns)
                    uncovered_yarns = bom_yarns - yarn_ids
                    
                    kpis.update({
                        'bom_coverage_count': len(covered_yarns),
                        'bom_missing_count': len(uncovered_yarns),
                        'bom_coverage_percentage': float(len(covered_yarns) / len(bom_yarns) * 100)
                    })
            
        except Exception as e:
            self.logger.error(f"Error calculating planning KPIs: {e}")
            kpis['planning_calculation_error'] = str(e)
        
        return kpis


class MultiLevelBOMNetting:
    """
    Complete multi-level Bill of Materials netting logic
    Handles complex BOM structures with proper explosion and netting
    """
    
    def __init__(self, path_resolver: DynamicPathResolver, column_system: ColumnAliasSystem):
        """Initialize with data access components"""
        self.logger = logging.getLogger(f'{__name__}.MultiLevelBOMNetting')
        self.path_resolver = path_resolver
        self.column_system = column_system
        
        # BOM explosion cache
        self._bom_cache = {}
        self._explosion_cache = {}
    
    def calculate_net_requirements(self, style_demands: Dict[str, float], 
                                 force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calculate net material requirements using multi-level BOM explosion
        
        Args:
            style_demands: Dictionary mapping style IDs to demanded quantities
            force_refresh: Whether to refresh cached BOM data
            
        Returns:
            Dictionary containing net requirements analysis
        """
        try:
            self.logger.info(f"Calculating net requirements for {len(style_demands)} styles")
            
            # Load BOM and inventory data
            bom_df = self._load_bom_data(force_refresh)
            inventory_df = self._load_inventory_data(force_refresh)
            
            if bom_df is None or inventory_df is None:
                raise ValueError("Could not load required BOM or inventory data")
            
            # Build BOM hierarchy
            bom_structure = self._build_bom_structure(bom_df)
            
            # Calculate gross requirements through BOM explosion
            gross_requirements = self._explode_bom_requirements(
                style_demands, bom_structure, levels=10  # Max 10 levels deep
            )
            
            # Calculate net requirements against available inventory
            net_requirements = self._calculate_net_against_inventory(
                gross_requirements, inventory_df
            )
            
            # Generate procurement plan
            procurement_plan = self._generate_procurement_plan(net_requirements, inventory_df)
            
            # Create summary report
            summary = self._create_netting_summary(
                style_demands, gross_requirements, net_requirements, procurement_plan
            )
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'calculation_status': 'success',
                'styles_processed': len(style_demands),
                'materials_analyzed': len(gross_requirements),
                'summary': summary,
                'gross_requirements': gross_requirements,
                'net_requirements': net_requirements,
                'procurement_plan': procurement_plan
            }
            
            self.logger.info(f"Net requirements calculation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in net requirements calculation: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'calculation_status': 'error',
                'error_message': str(e),
                'styles_processed': 0,
                'materials_analyzed': 0
            }
    
    def _load_bom_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Load and cache BOM data"""
        cache_key = 'bom_data'
        
        if not force_refresh and cache_key in self._bom_cache:
            return self._bom_cache[cache_key]
        
        bom_path = self.path_resolver.resolve_data_file('bom')
        if not bom_path:
            self.logger.error("Could not find BOM file")
            return None
        
        try:
            if bom_path.endswith('.csv'):
                bom_df = pd.read_csv(bom_path)
            else:
                bom_df = pd.read_excel(bom_path)
            
            # Standardize columns
            bom_df = self.column_system.standardize_columns(bom_df)
            
            # Validate required columns
            required_columns = ['style_id', 'yarn_id', 'bom_percentage']
            is_valid, missing = self.column_system.validate_required_columns(bom_df, required_columns)
            
            if not is_valid:
                self.logger.warning(f"BOM missing required columns: {missing}")
                # Try to find columns with different names
                for col_type in missing:
                    found_col = self.column_system.find_column(bom_df, col_type)
                    if found_col:
                        self.logger.info(f"Found {col_type} as '{found_col}'")
            
            # Clean and validate data
            bom_df = self._clean_bom_data(bom_df)
            
            self._bom_cache[cache_key] = bom_df
            self.logger.info(f"Loaded BOM data: {len(bom_df)} entries")
            
            return bom_df
            
        except Exception as e:
            self.logger.error(f"Error loading BOM data from {bom_path}: {e}")
            return None
    
    def _load_inventory_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Load and cache inventory data"""
        cache_key = 'inventory_data'
        
        if not force_refresh and cache_key in self._bom_cache:
            return self._bom_cache[cache_key]
        
        inventory_path = self.path_resolver.resolve_data_file('yarn_inventory')
        if not inventory_path:
            self.logger.error("Could not find yarn inventory file")
            return None
        
        try:
            if inventory_path.endswith('.csv'):
                inventory_df = pd.read_csv(inventory_path)
            else:
                inventory_df = pd.read_excel(inventory_path)
            
            # Standardize columns
            inventory_df = self.column_system.standardize_columns(inventory_df)
            
            # Clean inventory data
            inventory_df = self._clean_inventory_data(inventory_df)
            
            self._bom_cache[cache_key] = inventory_df
            self.logger.info(f"Loaded inventory data: {len(inventory_df)} items")
            
            return inventory_df
            
        except Exception as e:
            self.logger.error(f"Error loading inventory data from {inventory_path}: {e}")
            return None
    
    def _clean_bom_data(self, bom_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate BOM data"""
        # Find actual column names
        style_col = self.column_system.find_column(bom_df, 'style_id')
        yarn_col = self.column_system.find_column(bom_df, 'yarn_id')
        pct_col = self.column_system.find_column(bom_df, 'bom_percentage')
        
        if not all([style_col, yarn_col, pct_col]):
            self.logger.warning("Some BOM columns not found, using available columns")
        
        # Remove rows with missing critical data
        if style_col:
            bom_df = bom_df.dropna(subset=[style_col])
        if yarn_col:
            bom_df = bom_df.dropna(subset=[yarn_col])
        if pct_col:
            bom_df = bom_df.dropna(subset=[pct_col])
        
        # Convert percentage to numeric
        if pct_col:
            bom_df[pct_col] = pd.to_numeric(bom_df[pct_col], errors='coerce')
            
            # Handle percentage formats (both 0.15 and 15% formats)
            pct_values = bom_df[pct_col].copy()
            # If values are > 1, assume they're in percentage format (15% = 15)
            high_values_mask = pct_values > 1
            pct_values.loc[high_values_mask] = pct_values.loc[high_values_mask] / 100
            bom_df[pct_col] = pct_values
        
        # Remove invalid percentages
        if pct_col:
            bom_df = bom_df[(bom_df[pct_col] > 0) & (bom_df[pct_col] <= 1)]
        
        return bom_df
    
    def _clean_inventory_data(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate inventory data"""
        # Find actual column names
        yarn_col = self.column_system.find_column(inventory_df, 'yarn_id')
        balance_col = self.column_system.find_column(inventory_df, 'planning_balance')
        allocated_col = self.column_system.find_column(inventory_df, 'allocated')
        on_order_col = self.column_system.find_column(inventory_df, 'on_order')
        
        # Remove rows with missing yarn IDs
        if yarn_col:
            inventory_df = inventory_df.dropna(subset=[yarn_col])
        
        # Convert numeric columns
        for col, col_name in [(balance_col, 'balance'), (allocated_col, 'allocated'), (on_order_col, 'on_order')]:
            if col:
                inventory_df[col] = pd.to_numeric(inventory_df[col], errors='coerce').fillna(0)
        
        return inventory_df
    
    def _build_bom_structure(self, bom_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Build hierarchical BOM structure"""
        bom_structure = defaultdict(list)
        
        style_col = self.column_system.find_column(bom_df, 'style_id')
        yarn_col = self.column_system.find_column(bom_df, 'yarn_id')
        pct_col = self.column_system.find_column(bom_df, 'bom_percentage')
        
        if not all([style_col, yarn_col, pct_col]):
            self.logger.error("Required BOM columns not found")
            return {}
        
        for _, row in bom_df.iterrows():
            style_id = str(row[style_col]).strip()
            yarn_id = str(row[yarn_col]).strip()
            percentage = float(row[pct_col])
            
            bom_structure[style_id].append({
                'material_id': yarn_id,
                'percentage': percentage,
                'level': 1  # Direct material (level 1)
            })
        
        self.logger.info(f"Built BOM structure for {len(bom_structure)} styles")
        return dict(bom_structure)
    
    def _explode_bom_requirements(self, style_demands: Dict[str, float], 
                                 bom_structure: Dict[str, List[Dict]],
                                 levels: int = 10) -> Dict[str, float]:
        """Explode BOM to calculate gross material requirements"""
        gross_requirements = defaultdict(float)
        
        def explode_style(style_id: str, quantity: float, current_level: int = 0):
            """Recursively explode a style's BOM"""
            if current_level >= levels:
                self.logger.warning(f"Maximum BOM explosion depth reached for {style_id}")
                return
            
            if style_id not in bom_structure:
                self.logger.debug(f"No BOM found for style {style_id}")
                return
            
            for component in bom_structure[style_id]:
                material_id = component['material_id']
                percentage = component['percentage']
                required_qty = quantity * percentage
                
                gross_requirements[material_id] += required_qty
                
                # Check if this material is also a style (sub-assembly)
                if material_id in bom_structure:
                    explode_style(material_id, required_qty, current_level + 1)
        
        # Explode all demanded styles
        for style_id, demand_qty in style_demands.items():
            explode_style(style_id, demand_qty)
        
        self.logger.info(f"BOM explosion complete: {len(gross_requirements)} materials required")
        return dict(gross_requirements)
    
    def _calculate_net_against_inventory(self, gross_requirements: Dict[str, float],
                                       inventory_df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate net requirements after considering available inventory"""
        net_requirements = {}
        
        yarn_col = self.column_system.find_column(inventory_df, 'yarn_id')
        balance_col = self.column_system.find_column(inventory_df, 'planning_balance')
        allocated_col = self.column_system.find_column(inventory_df, 'allocated')
        on_order_col = self.column_system.find_column(inventory_df, 'on_order')
        
        if not yarn_col:
            self.logger.error("Cannot find yarn ID column in inventory")
            return {}
        
        # Create inventory lookup
        inventory_lookup = {}
        for _, row in inventory_df.iterrows():
            yarn_id = str(row[yarn_col]).strip()
            
            inventory_lookup[yarn_id] = {
                'planning_balance': float(row[balance_col]) if balance_col else 0.0,
                'allocated': float(row[allocated_col]) if allocated_col else 0.0,
                'on_order': float(row[on_order_col]) if on_order_col else 0.0
            }
        
        # Calculate net requirements
        for material_id, gross_qty in gross_requirements.items():
            material_id = str(material_id).strip()
            
            if material_id in inventory_lookup:
                inv_data = inventory_lookup[material_id]
                
                available_qty = inv_data['planning_balance']
                allocated_qty = inv_data['allocated']
                on_order_qty = inv_data['on_order']
                
                # Available for use = Planning Balance - Already Allocated
                usable_qty = available_qty - allocated_qty
                
                # Net requirement = Gross Requirement - Usable Inventory
                net_qty = gross_qty - usable_qty
                
                net_requirements[material_id] = {
                    'gross_requirement': gross_qty,
                    'available_inventory': available_qty,
                    'allocated_inventory': allocated_qty,
                    'usable_inventory': usable_qty,
                    'on_order_qty': on_order_qty,
                    'net_requirement': max(0, net_qty),  # Can't be negative
                    'surplus': max(0, -net_qty),  # Positive if we have surplus
                    'shortage': max(0, net_qty)   # Positive if we need to procure
                }
            else:
                # Material not in inventory - need to procure all
                net_requirements[material_id] = {
                    'gross_requirement': gross_qty,
                    'available_inventory': 0.0,
                    'allocated_inventory': 0.0,
                    'usable_inventory': 0.0,
                    'on_order_qty': 0.0,
                    'net_requirement': gross_qty,
                    'surplus': 0.0,
                    'shortage': gross_qty
                }
        
        return net_requirements
    
    def _generate_procurement_plan(self, net_requirements: Dict[str, Dict],
                                 inventory_df: pd.DataFrame) -> List[Dict]:
        """Generate procurement plan for materials with shortages"""
        procurement_plan = []
        
        for material_id, req_data in net_requirements.items():
            shortage = req_data['shortage']
            
            if shortage > 0:  # Only items that need procurement
                procurement_plan.append({
                    'material_id': material_id,
                    'shortage_qty': shortage,
                    'gross_requirement': req_data['gross_requirement'],
                    'available_inventory': req_data['available_inventory'],
                    'priority': self._calculate_priority(material_id, req_data),
                    'procurement_suggestion': shortage * 1.1  # Add 10% buffer
                })
        
        # Sort by priority (highest first)
        procurement_plan.sort(key=lambda x: x['priority'], reverse=True)
        
        return procurement_plan
    
    def _calculate_priority(self, material_id: str, req_data: Dict) -> float:
        """Calculate procurement priority score"""
        # Higher shortage = higher priority
        shortage_score = req_data['shortage'] / max(req_data['gross_requirement'], 1)
        
        # Materials with zero inventory get higher priority
        inventory_score = 1.0 if req_data['available_inventory'] <= 0 else 0.5
        
        # Combine scores
        priority = (shortage_score * 0.7) + (inventory_score * 0.3)
        
        return priority
    
    def _create_netting_summary(self, style_demands: Dict[str, float],
                              gross_requirements: Dict[str, float],
                              net_requirements: Dict[str, Dict],
                              procurement_plan: List[Dict]) -> Dict[str, Any]:
        """Create summary of netting calculation"""
        total_gross = sum(gross_requirements.values())
        total_net = sum(req['net_requirement'] for req in net_requirements.values())
        total_shortage = sum(req['shortage'] for req in net_requirements.values())
        total_surplus = sum(req['surplus'] for req in net_requirements.values())
        
        materials_with_shortage = sum(1 for req in net_requirements.values() if req['shortage'] > 0)
        materials_with_surplus = sum(1 for req in net_requirements.values() if req['surplus'] > 0)
        
        return {
            'styles_demanded': len(style_demands),
            'total_style_demand': sum(style_demands.values()),
            'materials_required': len(gross_requirements),
            'total_gross_requirement': total_gross,
            'total_net_requirement': total_net,
            'total_shortage': total_shortage,
            'total_surplus': total_surplus,
            'materials_with_shortage': materials_with_shortage,
            'materials_with_surplus': materials_with_surplus,
            'procurement_items_count': len(procurement_plan),
            'inventory_utilization_rate': (total_gross - total_net) / total_gross if total_gross > 0 else 0.0
        }


# Main integration functions for the ERP system
def initialize_emergency_fixes():
    """
    Initialize all emergency fix components
    
    Returns:
        Dictionary containing initialized components
    """
    logger.info("Initializing Day 0 Emergency Fixes")
    
    components = {
        'path_resolver': DynamicPathResolver(),
        'column_system': ColumnAliasSystem(),
        'price_parser': PriceStringParser()
    }
    
    # Initialize KPI calculator and BOM netting with dependencies
    components['kpi_calculator'] = RealKPICalculator(
        components['path_resolver'], 
        components['column_system']
    )
    
    components['bom_netting'] = MultiLevelBOMNetting(
        components['path_resolver'],
        components['column_system']
    )
    
    logger.info("Emergency fixes initialization complete")
    return components


def run_comprehensive_health_check():
    """
    Run a comprehensive health check of all emergency fixes
    
    Returns:
        Dictionary containing health check results
    """
    logger.info("Running comprehensive health check")
    
    try:
        # Initialize components
        components = initialize_emergency_fixes()
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components_tested': 0,
            'components_passed': 0,
            'issues_found': []
        }
        
        # Test path resolution
        logger.info("Testing path resolution...")
        resolved_files = components['path_resolver'].resolve_all_files()
        health_results['path_resolution'] = {
            'files_found': len(resolved_files),
            'total_file_types': len(components['path_resolver'].file_patterns),
            'success_rate': len(resolved_files) / len(components['path_resolver'].file_patterns)
        }
        
        if len(resolved_files) < len(components['path_resolver'].file_patterns) * 0.5:
            health_results['issues_found'].append("Low file resolution success rate")
        
        # Test data integrity
        logger.info("Testing data integrity...")
        validation_results = components['path_resolver'].validate_data_integrity(resolved_files)
        valid_files = sum(1 for result in validation_results.values() if result.get('format_valid', False))
        health_results['data_integrity'] = {
            'files_validated': len(validation_results),
            'valid_files': valid_files,
            'validation_success_rate': valid_files / len(validation_results) if validation_results else 0
        }
        
        # Test price parsing
        logger.info("Testing price parsing...")
        test_prices = ['$4.07', '$14.95 (kg)', '$1,234.56', '15.50', '€100.50', 'invalid']
        parsed_prices = components['price_parser'].parse_batch(test_prices)
        price_validation = components['price_parser'].validate_price_range(parsed_prices)
        health_results['price_parsing'] = price_validation
        
        # Test KPI calculation
        logger.info("Testing KPI calculation...")
        try:
            kpis = components['kpi_calculator'].calculate_comprehensive_kpis()
            health_results['kpi_calculation'] = {
                'status': kpis.get('calculation_status', 'unknown'),
                'data_sources_loaded': kpis.get('data_sources_loaded', 0),
                'kpis_calculated': len([k for k in kpis.keys() if not k.endswith('_error')])
            }
        except Exception as e:
            health_results['kpi_calculation'] = {
                'status': 'error',
                'error': str(e)
            }
            health_results['issues_found'].append(f"KPI calculation error: {e}")
        
        # Calculate overall health
        components_tested = 4
        components_passed = 0
        
        if health_results['path_resolution']['success_rate'] > 0.5:
            components_passed += 1
        if health_results['data_integrity']['validation_success_rate'] > 0.5:
            components_passed += 1
        if health_results['price_parsing']['success_rate'] > 0.8:
            components_passed += 1
        if health_results['kpi_calculation']['status'] == 'success':
            components_passed += 1
        
        health_results.update({
            'components_tested': components_tested,
            'components_passed': components_passed,
            'overall_health_score': components_passed / components_tested,
            'overall_status': 'healthy' if components_passed >= components_tested * 0.75 else 'degraded'
        })
        
        logger.info(f"Health check complete: {components_passed}/{components_tested} components healthy")
        return health_results
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'critical',
            'error_message': str(e),
            'components_tested': 0,
            'components_passed': 0
        }


if __name__ == "__main__":
    """
    Command-line interface for testing emergency fixes
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Beverly Knits ERP Day 0 Emergency Fixes')
    parser.add_argument('--health-check', action='store_true', 
                       help='Run comprehensive health check')
    parser.add_argument('--test-paths', action='store_true',
                       help='Test path resolution only')
    parser.add_argument('--test-prices', action='store_true',
                       help='Test price parsing only')
    parser.add_argument('--calculate-kpis', action='store_true',
                       help='Calculate KPIs only')
    parser.add_argument('--test-netting', action='store_true',
                       help='Test BOM netting with sample data')
    
    args = parser.parse_args()
    
    if args.health_check:
        results = run_comprehensive_health_check()
        print(json.dumps(results, indent=2))
    
    elif args.test_paths:
        resolver = DynamicPathResolver()
        files = resolver.resolve_all_files()
        print(f"Found {len(files)} files:")
        for file_type, path in files.items():
            print(f"  {file_type}: {path}")
    
    elif args.test_prices:
        parser = PriceStringParser()
        test_prices = [
            '$4.07', '$14.95 (kg)', '$1,234.56', '€100.50', 
            '15.50/lb', '10-20', '1.5E+3', 'invalid'
        ]
        for price in test_prices:
            result = parser.parse_price(price)
            print(f"{price} -> {result['value']} ({result['is_valid']})")
    
    elif args.calculate_kpis:
        components = initialize_emergency_fixes()
        kpis = components['kpi_calculator'].calculate_comprehensive_kpis()
        print(json.dumps(kpis, indent=2, default=str))
    
    elif args.test_netting:
        components = initialize_emergency_fixes()
        sample_demands = {'STYLE001': 100.0, 'STYLE002': 50.0}
        result = components['bom_netting'].calculate_net_requirements(sample_demands)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print("Day 0 Emergency Fixes for Beverly Knits ERP")
        print("Use --health-check to run comprehensive tests")
        print("Use --help to see all options")