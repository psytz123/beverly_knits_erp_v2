#!/usr/bin/env python3
"""
Data Validation Rules - Define and enforce data integrity rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidationRules:
    """Define and enforce validation rules for data consistency"""
    
    @staticmethod
    def validate_yarn_inventory(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate yarn inventory data
        
        Returns:
            Validation results with errors and warnings
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        # Required columns
        required_columns = ['Desc#', 'Planning Balance', 'Allocated']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['errors'].append({
                'type': 'MISSING_COLUMNS',
                'message': f"Missing required columns: {missing_columns}",
                'columns': missing_columns
            })
            results['is_valid'] = False
        
        # Check for duplicate yarn IDs
        if 'Desc#' in df.columns:
            duplicates = df[df.duplicated(subset=['Desc#'], keep=False)]
            if not duplicates.empty:
                results['warnings'].append({
                    'type': 'DUPLICATE_YARNS',
                    'message': f"Found {len(duplicates)} duplicate yarn entries",
                    'yarn_ids': duplicates['Desc#'].unique().tolist()[:10]
                })
        
        # Validate numeric columns
        numeric_columns = ['Planning Balance', 'Allocated', 'On Order', 'Theoretical Balance']
        for col in numeric_columns:
            if col in df.columns:
                # Check for non-numeric values
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()]
                    if not non_numeric.empty:
                        results['warnings'].append({
                            'type': 'NON_NUMERIC_VALUES',
                            'message': f"Column '{col}' contains non-numeric values",
                            'row_count': len(non_numeric)
                        })
                except Exception as e:
                    logger.error(f"Error validating column {col}: {e}")
        
        # Check Planning Balance consistency
        if all(col in df.columns for col in ['Planning Balance', 'Theoretical Balance', 'Allocated']):
            # Planning Balance should equal Theoretical Balance + Allocated
            df['calculated_planning'] = df['Theoretical Balance'] + df['Allocated']
            discrepancies = df[abs(df['Planning Balance'] - df['calculated_planning']) > 0.01]
            
            if not discrepancies.empty:
                results['warnings'].append({
                    'type': 'BALANCE_DISCREPANCY',
                    'message': f"Planning Balance doesn't match Theoretical + Allocated for {len(discrepancies)} yarns",
                    'sample_yarns': discrepancies['Desc#'].head(5).tolist() if 'Desc#' in df.columns else []
                })
        
        return results
    
    @staticmethod
    def validate_bom(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate BOM data
        
        Returns:
            Validation results with errors and warnings
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'statistics': {}
        }
        
        # Required columns
        required_columns = ['Style#', 'Desc#', 'BOM_Percentage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['errors'].append({
                'type': 'MISSING_COLUMNS',
                'message': f"Missing required columns: {missing_columns}",
                'columns': missing_columns
            })
            results['is_valid'] = False
            return results
        
        # Check BOM percentages sum to 1 (or 100%) per style
        style_totals = df.groupby('Style#')['BOM_Percentage'].sum()
        
        # Check if percentages are in decimal (0-1) or percentage (0-100) format
        max_percentage = df['BOM_Percentage'].max()
        if max_percentage > 1:
            # Likely in percentage format (0-100)
            invalid_styles = style_totals[(style_totals < 99) | (style_totals > 101)]
            percentage_format = 'percentage'
        else:
            # Decimal format (0-1)
            invalid_styles = style_totals[(style_totals < 0.99) | (style_totals > 1.01)]
            percentage_format = 'decimal'
        
        if not invalid_styles.empty:
            results['warnings'].append({
                'type': 'INVALID_BOM_TOTAL',
                'message': f"{len(invalid_styles)} styles have BOM percentages that don't sum to 100%",
                'format': percentage_format,
                'sample_styles': invalid_styles.head(10).to_dict()
            })
        
        # Check for styles with no BOM entries
        if 'Style#' in df.columns:
            unique_styles = df['Style#'].nunique()
            results['statistics']['unique_styles'] = unique_styles
            results['statistics']['total_mappings'] = len(df)
            results['statistics']['avg_yarns_per_style'] = len(df) / unique_styles if unique_styles > 0 else 0
        
        # Check for invalid yarn references
        if 'Desc#' in df.columns:
            invalid_yarns = df[df['Desc#'].isna() | (df['Desc#'] == '')]
            if not invalid_yarns.empty:
                results['warnings'].append({
                    'type': 'INVALID_YARN_REFERENCE',
                    'message': f"{len(invalid_yarns)} BOM entries have invalid yarn references",
                    'affected_styles': invalid_yarns['Style#'].unique().tolist()[:10]
                })
        
        return results
    
    @staticmethod
    def validate_production_orders(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate production orders data
        
        Returns:
            Validation results with errors and warnings
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'statistics': {}
        }
        
        # Required columns
        required_columns = ['Style #', 'Order #', 'Qty Ordered (lbs)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try alternate column names
            alt_required = ['Style#', 'Order#', 'Quantity']
            alt_missing = [col for col in alt_required if col not in df.columns]
            
            if alt_missing:
                results['errors'].append({
                    'type': 'MISSING_COLUMNS',
                    'message': f"Missing required columns: {missing_columns}",
                    'columns': missing_columns,
                    'alternates_tried': alt_required
                })
                results['is_valid'] = False
        
        # Check for duplicate order numbers
        order_col = 'Order #' if 'Order #' in df.columns else 'Order#' if 'Order#' in df.columns else None
        if order_col:
            duplicates = df[df.duplicated(subset=[order_col], keep=False)]
            if not duplicates.empty:
                results['errors'].append({
                    'type': 'DUPLICATE_ORDERS',
                    'message': f"Found {len(duplicates)} duplicate order numbers",
                    'order_numbers': duplicates[order_col].unique().tolist()[:10]
                })
                results['is_valid'] = False
        
        # Validate quantities
        qty_columns = ['Qty Ordered (lbs)', 'Balance (lbs)', 'G00 (lbs)', 'Shipped (lbs)']
        for col in qty_columns:
            if col in df.columns:
                # Parse numeric values (handle commas)
                df[f'{col}_numeric'] = df[col].apply(lambda x: 
                    float(str(x).replace(',', '')) if pd.notna(x) and str(x).replace(',', '').replace('.', '').isdigit() else 0
                )
                
                negative_values = df[df[f'{col}_numeric'] < 0]
                if not negative_values.empty:
                    results['warnings'].append({
                        'type': 'NEGATIVE_QUANTITIES',
                        'message': f"Column '{col}' contains negative values",
                        'row_count': len(negative_values)
                    })
        
        # Check balance consistency
        if all(col in df.columns for col in ['Qty Ordered (lbs)', 'Shipped (lbs)', 'Balance (lbs)']):
            # Balance should equal Ordered - Shipped
            df['calc_balance'] = df['Qty Ordered (lbs)_numeric'] - df['Shipped (lbs)_numeric'].fillna(0)
            discrepancies = df[abs(df['Balance (lbs)_numeric'] - df['calc_balance']) > 1]
            
            if not discrepancies.empty:
                results['warnings'].append({
                    'type': 'BALANCE_MISMATCH',
                    'message': f"Balance doesn't match Ordered - Shipped for {len(discrepancies)} orders",
                    'sample_orders': discrepancies[order_col].head(5).tolist() if order_col else []
                })
        
        # Statistics
        if 'Machine' in df.columns:
            assigned_orders = df[df['Machine'].notna() & (df['Machine'] != '')]
            results['statistics']['assigned_orders'] = len(assigned_orders)
            results['statistics']['unassigned_orders'] = len(df) - len(assigned_orders)
            results['statistics']['unique_machines'] = df['Machine'].nunique()
        
        return results
    
    @staticmethod
    def cross_validate_data(inventory_df: pd.DataFrame,
                          bom_df: pd.DataFrame,
                          production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-validate data between different sources
        
        Returns:
            Cross-validation results
        """
        results = {
            'is_valid': True,
            'cross_checks': [],
            'missing_references': [],
            'orphaned_data': []
        }
        
        # Check 1: All production styles should have BOM entries
        if 'Style #' in production_df.columns and 'Style#' in bom_df.columns:
            production_styles = set(production_df['Style #'].unique())
            bom_styles = set(bom_df['Style#'].unique())
            
            missing_bom = production_styles - bom_styles
            if missing_bom:
                results['missing_references'].append({
                    'type': 'PRODUCTION_WITHOUT_BOM',
                    'message': f"{len(missing_bom)} production styles have no BOM",
                    'styles': list(missing_bom)[:20],
                    'impact': 'Cannot calculate yarn requirements for these styles'
                })
                results['is_valid'] = False
        
        # Check 2: All BOM yarns should exist in inventory
        if 'Desc#' in bom_df.columns and 'Desc#' in inventory_df.columns:
            bom_yarns = set(bom_df['Desc#'].unique())
            inventory_yarns = set(inventory_df['Desc#'].unique())
            
            missing_yarns = bom_yarns - inventory_yarns
            if missing_yarns:
                results['missing_references'].append({
                    'type': 'BOM_YARN_NOT_IN_INVENTORY',
                    'message': f"{len(missing_yarns)} BOM yarns not found in inventory",
                    'yarns': list(missing_yarns)[:20],
                    'impact': 'Cannot check availability for these yarns'
                })
        
        # Check 3: Orphaned inventory (yarns not used in any BOM)
        if 'Desc#' in bom_df.columns and 'Desc#' in inventory_df.columns:
            orphaned_yarns = inventory_yarns - bom_yarns
            if orphaned_yarns:
                # Check if these yarns have inventory
                orphaned_with_inventory = inventory_df[
                    (inventory_df['Desc#'].isin(orphaned_yarns)) & 
                    (inventory_df['Planning Balance'] > 0)
                ]
                
                if not orphaned_with_inventory.empty:
                    results['orphaned_data'].append({
                        'type': 'UNUSED_YARN_INVENTORY',
                        'message': f"{len(orphaned_with_inventory)} yarns have inventory but no BOM usage",
                        'total_value': orphaned_with_inventory['Planning Balance'].sum(),
                        'sample_yarns': orphaned_with_inventory['Desc#'].head(10).tolist()
                    })
        
        return results