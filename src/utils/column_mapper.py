"""
Column Mapper
Unified column mapping for consistent data handling across the system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """Column mapping configuration"""
    standard_name: str
    variations: List[str]
    data_type: str
    required: bool = False
    default_value: Any = None
    validation_regex: Optional[str] = None
    transformation: Optional[str] = None


class ColumnMapper:
    """Unified column mapping for consistent data handling"""
    
    # Master mapping configuration
    COLUMN_MAPPINGS = {
        'planning_balance': ColumnMapping(
            standard_name='planning_balance',
            variations=['Planning Balance', 'Planning_Balance', 'planning balance', 
                       'PlanningBalance', 'plan_balance', 'Plan Balance'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'yarn_id': ColumnMapping(
            standard_name='yarn_id',
            variations=['Desc#', 'desc_num', 'YarnID', 'yarn_id', 'Yarn ID', 
                       'Yarn_ID', 'yarn', 'YARN_ID', 'desc_no', 'Description#'],
            data_type='str',
            required=True
        ),
        'style_id': ColumnMapping(
            standard_name='style_id',
            variations=['fStyle#', 'Style#', 'style_num', 'Style', 'StyleID',
                       'style_id', 'Style_ID', 'STYLE_ID', 'style_no', 'StyleNum'],
            data_type='str',
            required=False
        ),
        'quantity': ColumnMapping(
            standard_name='quantity',
            variations=['Qty', 'Quantity', 'quantity', 'Amount', 'QTY', 
                       'qty', 'amount', 'Quant', 'quan'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'balance': ColumnMapping(
            standard_name='balance',
            variations=['Balance (lbs)', 'Balance', 'balance_lbs', 'Balance_lbs',
                       'Bal', 'balance', 'BAL', 'Balance(lbs)'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'theoretical_balance': ColumnMapping(
            standard_name='theoretical_balance',
            variations=['Theoretical Balance', 'Theoretical_Balance', 'theoretical_balance',
                       'Theo Balance', 'theo_balance', 'TheoreticalBalance'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'allocated': ColumnMapping(
            standard_name='allocated',
            variations=['Allocated', 'allocated', 'Alloc', 'alloc', 'ALLOCATED',
                       'Allocation', 'allocation'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'on_order': ColumnMapping(
            standard_name='on_order',
            variations=['On Order', 'On_Order', 'on_order', 'OnOrder', 'PO Qty',
                       'po_qty', 'PO_Qty', 'Purchase Order', 'purchase_order'],
            data_type='float',
            required=False,
            default_value=0.0
        ),
        'work_center': ColumnMapping(
            standard_name='work_center',
            variations=['Work Center', 'Work_Center', 'work_center', 'WorkCenter',
                       'WC', 'wc', 'W/C', 'Work_Ctr', 'work_ctr'],
            data_type='str',
            required=False
        ),
        'machine_id': ColumnMapping(
            standard_name='machine_id',
            variations=['Machine', 'Machine ID', 'Machine_ID', 'machine_id', 
                       'MachineID', 'MACH', 'mach', 'Machine#', 'machine_no'],
            data_type='str',
            required=False
        ),
        'order_id': ColumnMapping(
            standard_name='order_id',
            variations=['Order ID', 'Order_ID', 'order_id', 'OrderID', 'Order#',
                       'order_no', 'Order No', 'PO#', 'po_no', 'Purchase Order#'],
            data_type='str',
            required=False
        ),
        'date': ColumnMapping(
            standard_name='date',
            variations=['Date', 'date', 'DATE', 'DateTime', 'datetime', 
                       'Timestamp', 'timestamp', 'Created', 'created_date'],
            data_type='datetime',
            required=False
        ),
        'scheduled_date': ColumnMapping(
            standard_name='scheduled_date',
            variations=['Scheduled Date', 'Scheduled_Date', 'scheduled_date',
                       'Schedule Date', 'schedule_date', 'Due Date', 'due_date'],
            data_type='datetime',
            required=False
        ),
        'status': ColumnMapping(
            standard_name='status',
            variations=['Status', 'status', 'STATUS', 'State', 'state',
                       'Order Status', 'order_status'],
            data_type='str',
            required=False,
            validation_regex=r'^(pending|scheduled|in_progress|completed|cancelled)$'
        ),
        'unit_price': ColumnMapping(
            standard_name='unit_price',
            variations=['Unit Price', 'Unit_Price', 'unit_price', 'UnitPrice',
                       'Price', 'price', 'Cost', 'cost', 'Unit Cost'],
            data_type='float',
            required=False,
            transformation='remove_currency'
        ),
        'description': ColumnMapping(
            standard_name='description',
            variations=['Description', 'description', 'DESC', 'desc', 'Desc',
                       'Name', 'name', 'Item Name', 'item_name'],
            data_type='str',
            required=False
        ),
        'supplier': ColumnMapping(
            standard_name='supplier',
            variations=['Supplier', 'supplier', 'SUPPLIER', 'Vendor', 'vendor',
                       'Supplier Name', 'supplier_name', 'Vendor Name'],
            data_type='str',
            required=False
        ),
        'category': ColumnMapping(
            standard_name='category',
            variations=['Category', 'category', 'CAT', 'cat', 'Type', 'type',
                       'Item Category', 'item_category', 'Product Category'],
            data_type='str',
            required=False
        )
    }
    
    @classmethod
    def standardize_dataframe(cls, 
                            df: pd.DataFrame,
                            strict: bool = False) -> pd.DataFrame:
        """
        Standardize all column names in DataFrame
        
        Args:
            df: DataFrame to standardize
            strict: If True, only keep mapped columns
            
        Returns:
            DataFrame with standardized column names
        """
        if df.empty:
            return df
        
        standardized_df = df.copy()
        mapped_columns = {}
        unmapped_columns = []
        
        # Map each column
        for col in df.columns:
            mapped = False
            
            for mapping_key, mapping in cls.COLUMN_MAPPINGS.items():
                if cls._matches_column(col, mapping.variations):
                    mapped_columns[col] = mapping.standard_name
                    mapped = True
                    
                    # Apply data type conversion
                    standardized_df = cls._convert_column_type(
                        standardized_df, 
                        col, 
                        mapping
                    )
                    break
            
            if not mapped:
                unmapped_columns.append(col)
                if not strict:
                    # Keep unmapped columns with cleaned names
                    mapped_columns[col] = cls._clean_column_name(col)
        
        # Rename columns
        standardized_df = standardized_df.rename(columns=mapped_columns)
        
        # Log unmapped columns if any
        if unmapped_columns:
            logger.debug(f"Unmapped columns: {unmapped_columns}")
        
        # Drop unmapped columns if strict mode
        if strict and unmapped_columns:
            standardized_df = standardized_df.drop(columns=unmapped_columns)
        
        return standardized_df
    
    @classmethod
    def _matches_column(cls, column: str, variations: List[str]) -> bool:
        """Check if column matches any variation"""
        column_clean = column.strip().lower()
        
        for variation in variations:
            variation_clean = variation.strip().lower()
            
            # Exact match
            if column_clean == variation_clean:
                return True
            
            # Match without spaces/underscores
            column_normalized = re.sub(r'[_\s]+', '', column_clean)
            variation_normalized = re.sub(r'[_\s]+', '', variation_clean)
            
            if column_normalized == variation_normalized:
                return True
        
        return False
    
    @classmethod
    def _clean_column_name(cls, column: str) -> str:
        """Clean column name to standard format"""
        # Remove special characters except underscore
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', column)
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    @classmethod
    def _convert_column_type(cls, 
                            df: pd.DataFrame,
                            column: str,
                            mapping: ColumnMapping) -> pd.DataFrame:
        """Convert column to specified data type"""
        if column not in df.columns:
            return df
        
        try:
            if mapping.data_type == 'float':
                # Handle comma-separated numbers and currency
                if df[column].dtype == 'object':
                    df[column] = df[column].astype(str).str.replace(',', '')
                    df[column] = df[column].str.replace('$', '')
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                else:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                
                # Fill NaN with default value
                if mapping.default_value is not None:
                    df[column] = df[column].fillna(mapping.default_value)
                    
            elif mapping.data_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = df[column].fillna(mapping.default_value or 0).astype(int)
                
            elif mapping.data_type == 'str':
                df[column] = df[column].astype(str)
                
                # Clean string values
                df[column] = df[column].str.strip()
                df[column] = df[column].replace('nan', '')
                
            elif mapping.data_type == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
                
            elif mapping.data_type == 'bool':
                df[column] = df[column].astype(bool)
            
            # Apply transformation if specified
            if mapping.transformation:
                df[column] = cls._apply_transformation(df[column], mapping.transformation)
            
            # Apply validation if specified
            if mapping.validation_regex:
                df[column] = cls._validate_column(df[column], mapping.validation_regex)
                
        except Exception as e:
            logger.warning(f"Error converting column {column} to {mapping.data_type}: {e}")
        
        return df
    
    @classmethod
    def _apply_transformation(cls, series: pd.Series, transformation: str) -> pd.Series:
        """Apply transformation to series"""
        if transformation == 'remove_currency':
            if series.dtype == 'object':
                series = series.str.replace(r'[$,]', '', regex=True)
                series = pd.to_numeric(series, errors='coerce')
        
        elif transformation == 'uppercase':
            if series.dtype == 'object':
                series = series.str.upper()
        
        elif transformation == 'lowercase':
            if series.dtype == 'object':
                series = series.str.lower()
        
        elif transformation == 'trim':
            if series.dtype == 'object':
                series = series.str.strip()
        
        return series
    
    @classmethod
    def _validate_column(cls, series: pd.Series, regex: str) -> pd.Series:
        """Validate column values against regex"""
        if series.dtype == 'object':
            # Mark invalid values as NaN
            mask = series.str.match(regex, na=False)
            series.loc[~mask] = np.nan
        
        return series
    
    @classmethod
    def find_column(cls, 
                   df: pd.DataFrame,
                   column_type: str) -> Optional[str]:
        """
        Find column by type, trying all variations
        
        Args:
            df: DataFrame to search
            column_type: Type of column to find
            
        Returns:
            Column name if found, None otherwise
        """
        if column_type not in cls.COLUMN_MAPPINGS:
            return None
        
        mapping = cls.COLUMN_MAPPINGS[column_type]
        
        for col in df.columns:
            if cls._matches_column(col, mapping.variations):
                return col
        
        return None
    
    @classmethod
    def find_all_columns(cls,
                        df: pd.DataFrame,
                        column_types: List[str]) -> Dict[str, Optional[str]]:
        """
        Find multiple columns by type
        
        Args:
            df: DataFrame to search
            column_types: List of column types to find
            
        Returns:
            Dictionary mapping column type to found column name
        """
        results = {}
        
        for column_type in column_types:
            results[column_type] = cls.find_column(df, column_type)
        
        return results
    
    @classmethod
    def validate_required_columns(cls,
                                df: pd.DataFrame,
                                required: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate DataFrame has required columns
        
        Args:
            df: DataFrame to validate
            required: List of required column types
            
        Returns:
            Tuple of (is_valid, validation_result)
        """
        result = {
            'valid': True,
            'missing': [],
            'found': {},
            'warnings': []
        }
        
        for req_col in required:
            col_name = cls.find_column(df, req_col)
            
            if col_name:
                result['found'][req_col] = col_name
                
                # Check if column has valid data
                if df[col_name].isna().all():
                    result['warnings'].append(f"Column {col_name} ({req_col}) contains only NaN values")
            else:
                result['missing'].append(req_col)
                result['valid'] = False
        
        return result['valid'], result
    
    @classmethod
    def get_mapping_report(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate mapping report for DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Mapping report
        """
        report = {
            'total_columns': len(df.columns),
            'mapped_columns': [],
            'unmapped_columns': [],
            'column_mappings': {},
            'data_types': {}
        }
        
        for col in df.columns:
            mapped = False
            
            for mapping_key, mapping in cls.COLUMN_MAPPINGS.items():
                if cls._matches_column(col, mapping.variations):
                    report['mapped_columns'].append(col)
                    report['column_mappings'][col] = {
                        'standard_name': mapping.standard_name,
                        'data_type': mapping.data_type,
                        'required': mapping.required
                    }
                    mapped = True
                    break
            
            if not mapped:
                report['unmapped_columns'].append(col)
            
            # Add data type info
            report['data_types'][col] = str(df[col].dtype)
        
        report['mapping_rate'] = len(report['mapped_columns']) / len(df.columns) if df.columns.any() else 0
        
        return report
    
    @classmethod
    def merge_dataframes_with_mapping(cls,
                                     dfs: List[pd.DataFrame],
                                     how: str = 'outer') -> pd.DataFrame:
        """
        Merge multiple DataFrames with automatic column mapping
        
        Args:
            dfs: List of DataFrames to merge
            how: Merge method ('outer', 'inner', 'left', 'right')
            
        Returns:
            Merged DataFrame with standardized columns
        """
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return cls.standardize_dataframe(dfs[0])
        
        # Standardize all DataFrames
        standardized_dfs = [cls.standardize_dataframe(df) for df in dfs]
        
        # Find common index columns
        common_cols = set(standardized_dfs[0].columns)
        for df in standardized_dfs[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Determine merge keys
        merge_keys = []
        for key in ['yarn_id', 'style_id', 'order_id']:
            if key in common_cols:
                merge_keys.append(key)
        
        if not merge_keys:
            # No common keys, concatenate instead
            return pd.concat(standardized_dfs, ignore_index=True)
        
        # Merge DataFrames
        result = standardized_dfs[0]
        for df in standardized_dfs[1:]:
            result = pd.merge(result, df, on=merge_keys, how=how, suffixes=('', '_dup'))
        
        # Remove duplicate columns
        dup_cols = [col for col in result.columns if col.endswith('_dup')]
        result = result.drop(columns=dup_cols)
        
        return result