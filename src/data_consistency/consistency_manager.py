#!/usr/bin/env python3
"""
Data Consistency Manager - Centralized logic for consistent data handling
Ensures all modules calculate shortages, demand, and aggregations the same way
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataConsistencyManager:
    """Centralized manager for data consistency across the ERP system"""
    
    # Standard column mappings - single source of truth
    COLUMN_MAPPINGS = {
        # Yarn inventory columns
        'yarn_id': ['Desc#', 'desc_num', 'yarn_id', 'YarnID', 'yarn_code'],
        'description': ['Description', 'description', 'yarn_description', 'desc'],
        'planning_balance': ['Planning Balance', 'Planning_Balance', 'planning_balance', 'PlanningBalance'],
        'theoretical_balance': ['Theoretical Balance', 'Theoretical_Balance', 'theoretical_balance'],
        'allocated': ['Allocated', 'allocated', 'Allocation'],
        'on_order': ['On Order', 'On_Order', 'on_order', 'OnOrder'],
        'beginning_balance': ['Beginning Balance', 'Beginning_Balance', 'beginning_balance'],
        
        # BOM columns
        'style_id': ['Style#', 'fStyle#', 'style_id', 'StyleID', 'style_code'],
        'bom_percentage': ['BOM_Percentage', 'bom_percentage', 'percentage', 'Percentage'],
        'unit': ['unit', 'Unit', 'units', 'UOM'],
        
        # Production columns
        'quantity_ordered': ['Qty Ordered (lbs)', 'Qty Ordered', 'quantity_ordered', 'qty_ordered'],
        'balance': ['Balance (lbs)', 'Balance', 'balance', 'remaining_balance'],
        'machine': ['Machine', 'machine', 'machine_id', 'MachineID'],
        'start_date': ['Start Date', 'start_date', 'StartDate'],
        'order_number': ['Order #', 'order_number', 'OrderNumber', 'order_num']
    }
    
    # Shortage thresholds
    SHORTAGE_THRESHOLDS = {
        'critical': -1000,  # Planning Balance < -1000 lbs
        'high': -500,       # Planning Balance < -500 lbs  
        'medium': -100,     # Planning Balance < -100 lbs
        'low': 0            # Planning Balance < 0 lbs
    }
    
    @classmethod
    def get_column_name(cls, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """
        Get the actual column name from dataframe for a given column type
        
        Args:
            df: DataFrame to check
            column_type: Type of column to find (e.g., 'yarn_id', 'planning_balance')
            
        Returns:
            Actual column name if found, None otherwise
        """
        if column_type not in cls.COLUMN_MAPPINGS:
            return None
            
        possible_names = cls.COLUMN_MAPPINGS[column_type]
        for name in possible_names:
            if name in df.columns:
                return name
        
        return None
    
    @classmethod
    def standardize_columns(cls, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Standardize column names to use consistent naming
        
        Args:
            df: DataFrame to standardize
            inplace: Whether to modify the dataframe in place
            
        Returns:
            DataFrame with standardized column names
        """
        if not inplace:
            df = df.copy()
        
        rename_map = {}
        for standard_name, possible_names in cls.COLUMN_MAPPINGS.items():
            for col in possible_names:
                if col in df.columns and col != standard_name:
                    rename_map[col] = standard_name
                    break
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.info(f"Standardized {len(rename_map)} column names")
        
        return df
    
    @classmethod
    def calculate_yarn_shortage(cls, yarn_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate shortage for a single yarn using consistent logic
        
        Args:
            yarn_row: Series containing yarn data
            
        Returns:
            Dictionary with shortage analysis
        """
        # Get values with fallback
        planning_balance = cls._get_value(yarn_row, 'planning_balance', 0)
        theoretical_balance = cls._get_value(yarn_row, 'theoretical_balance', planning_balance)
        allocated = abs(cls._get_value(yarn_row, 'allocated', 0))  # Allocated is typically negative
        on_order = cls._get_value(yarn_row, 'on_order', 0)
        
        # Calculate effective balance (planning balance is already net of allocated)
        effective_balance = planning_balance
        
        # Determine if there's a shortage - using OR logic as in main ERP
        has_shortage = (planning_balance < 0) or (theoretical_balance < 0)
        
        # Calculate shortage amount (use most negative value)
        shortage_amount = 0
        if has_shortage:
            shortage_amount = abs(min(planning_balance, theoretical_balance))
        
        # Determine risk level based on shortage amount
        risk_level = 'NONE'
        if shortage_amount > abs(cls.SHORTAGE_THRESHOLDS['critical']):
            risk_level = 'CRITICAL'
        elif shortage_amount > abs(cls.SHORTAGE_THRESHOLDS['high']):
            risk_level = 'HIGH'
        elif shortage_amount > abs(cls.SHORTAGE_THRESHOLDS['medium']):
            risk_level = 'MEDIUM'
        elif shortage_amount > 0:
            risk_level = 'LOW'
        
        # Calculate urgency score (0-100)
        urgency_score = min(100, (shortage_amount / abs(cls.SHORTAGE_THRESHOLDS['critical'])) * 100)
        
        return {
            'yarn_id': cls._get_value(yarn_row, 'yarn_id', ''),
            'description': cls._get_value(yarn_row, 'description', ''),
            'planning_balance': planning_balance,
            'theoretical_balance': theoretical_balance,
            'allocated': allocated,
            'on_order': on_order,
            'effective_balance': effective_balance,
            'has_shortage': has_shortage,
            'shortage_amount': shortage_amount,
            'risk_level': risk_level,
            'urgency_score': round(urgency_score, 1)
        }
    
    @classmethod
    def aggregate_yarn_requirements(cls, bom_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate yarn requirements from BOM and production orders
        
        Args:
            bom_df: BOM dataframe with style-yarn mappings
            production_df: Production orders dataframe
            
        Returns:
            Dictionary of yarn_id -> total required pounds
        """
        yarn_requirements = {}
        
        # Standardize columns
        bom_df = cls.standardize_columns(bom_df)
        production_df = cls.standardize_columns(production_df)
        
        # Get column names
        style_col = cls.get_column_name(bom_df, 'style_id')
        yarn_col = cls.get_column_name(bom_df, 'yarn_id')
        percentage_col = cls.get_column_name(bom_df, 'bom_percentage')
        
        prod_style_col = cls.get_column_name(production_df, 'style_id')
        qty_col = cls.get_column_name(production_df, 'quantity_ordered')
        balance_col = cls.get_column_name(production_df, 'balance')
        
        if not all([style_col, yarn_col, percentage_col, prod_style_col]):
            logger.warning("Missing required columns for BOM aggregation")
            return yarn_requirements
        
        # Use balance if available, otherwise use quantity ordered
        quantity_col = balance_col if balance_col else qty_col
        if not quantity_col:
            logger.warning("No quantity column found in production data")
            return yarn_requirements
        
        # Process each production order
        for _, order in production_df.iterrows():
            style = order.get(prod_style_col, '')
            if not style:
                continue
                
            # Get quantity (handle comma-separated numbers)
            quantity_str = str(order.get(quantity_col, 0))
            quantity = cls._parse_number(quantity_str)
            
            if quantity <= 0:
                continue
            
            # Find BOM entries for this style
            style_bom = bom_df[bom_df[style_col] == style]
            
            for _, bom_entry in style_bom.iterrows():
                yarn_id = str(bom_entry.get(yarn_col, ''))
                percentage = bom_entry.get(percentage_col, 0)
                
                if yarn_id and percentage > 0:
                    # Calculate yarn requirement for this order
                    yarn_needed = quantity * percentage
                    
                    # Aggregate
                    if yarn_id not in yarn_requirements:
                        yarn_requirements[yarn_id] = 0
                    yarn_requirements[yarn_id] += yarn_needed
        
        logger.info(f"Aggregated requirements for {len(yarn_requirements)} yarns")
        return yarn_requirements
    
    @classmethod
    def validate_data_consistency(cls, 
                                 inventory_df: pd.DataFrame,
                                 bom_df: pd.DataFrame,
                                 production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data consistency across different dataframes
        
        Returns:
            Dictionary with validation results and discrepancies
        """
        validation_results = {
            'is_consistent': True,
            'discrepancies': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Standardize all dataframes
        inventory_df = cls.standardize_columns(inventory_df)
        bom_df = cls.standardize_columns(bom_df)
        production_df = cls.standardize_columns(production_df)
        
        # Check 1: Yarn shortages consistency
        shortage_analysis = []
        yarn_col = cls.get_column_name(inventory_df, 'yarn_id')
        
        if yarn_col:
            for _, yarn in inventory_df.iterrows():
                shortage = cls.calculate_yarn_shortage(yarn)
                if shortage['has_shortage']:
                    shortage_analysis.append(shortage)
        
        validation_results['statistics']['total_shortages'] = len(shortage_analysis)
        validation_results['statistics']['critical_shortages'] = sum(1 for s in shortage_analysis if s['risk_level'] == 'CRITICAL')
        
        # Check 2: BOM coverage
        style_col_bom = cls.get_column_name(bom_df, 'style_id')
        style_col_prod = cls.get_column_name(production_df, 'style_id')
        
        if style_col_bom and style_col_prod:
            production_styles = set(production_df[style_col_prod].unique())
            bom_styles = set(bom_df[style_col_bom].unique())
            
            missing_bom = production_styles - bom_styles
            if missing_bom:
                validation_results['warnings'].append({
                    'type': 'MISSING_BOM',
                    'message': f"{len(missing_bom)} production styles have no BOM entries",
                    'styles': list(missing_bom)[:10]  # First 10 as sample
                })
                validation_results['is_consistent'] = False
        
        # Check 3: Yarn requirements vs inventory
        yarn_requirements = cls.aggregate_yarn_requirements(bom_df, production_df)
        
        for yarn_id, required in yarn_requirements.items():
            # Find yarn in inventory
            if yarn_col:
                yarn_inventory = inventory_df[inventory_df[yarn_col] == yarn_id]
                if not yarn_inventory.empty:
                    yarn_data = yarn_inventory.iloc[0]
                    planning_balance = cls._get_value(yarn_data, 'planning_balance', 0)
                    
                    # Check if shortage amount matches requirement
                    if planning_balance < 0 and abs(planning_balance) < required * 0.9:
                        validation_results['discrepancies'].append({
                            'type': 'SHORTAGE_MISMATCH',
                            'yarn_id': yarn_id,
                            'shortage_shown': abs(planning_balance),
                            'requirement_calculated': required,
                            'difference': required - abs(planning_balance)
                        })
        
        validation_results['statistics']['yarn_requirements_count'] = len(yarn_requirements)
        validation_results['statistics']['total_required_lbs'] = sum(yarn_requirements.values())
        
        return validation_results
    
    @classmethod
    def _get_value(cls, row: pd.Series, column_type: str, default: Any = None) -> Any:
        """Helper to get value from row with fallback"""
        possible_names = cls.COLUMN_MAPPINGS.get(column_type, [])
        for name in possible_names:
            if name in row.index and pd.notna(row[name]):
                return row[name]
        return default
    
    @classmethod
    def _parse_number(cls, value: Any) -> float:
        """Parse number from string, handling commas and currency"""
        if pd.isna(value):
            return 0.0
        
        str_value = str(value)
        # Remove currency symbols, commas, and spaces
        cleaned = str_value.replace('$', '').replace(',', '').replace(' ', '').strip()
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    
    @classmethod
    def create_reconciliation_report(cls,
                                    inventory_df: pd.DataFrame,
                                    bom_df: pd.DataFrame,
                                    production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a comprehensive reconciliation report
        
        Returns:
            Dictionary with reconciliation details and recommendations
        """
        # Validate consistency
        validation = cls.validate_data_consistency(inventory_df, bom_df, production_df)
        
        # Calculate shortages
        all_shortages = []
        inventory_df = cls.standardize_columns(inventory_df)
        
        for _, yarn in inventory_df.iterrows():
            shortage = cls.calculate_yarn_shortage(yarn)
            if shortage['has_shortage']:
                all_shortages.append(shortage)
        
        # Sort by urgency
        all_shortages.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        # Aggregate yarn requirements
        yarn_requirements = cls.aggregate_yarn_requirements(bom_df, production_df)
        
        # Build report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'validation': validation,
            'shortages': {
                'summary': {
                    'total_yarns_short': len(all_shortages),
                    'critical_count': sum(1 for s in all_shortages if s['risk_level'] == 'CRITICAL'),
                    'high_count': sum(1 for s in all_shortages if s['risk_level'] == 'HIGH'),
                    'total_shortage_lbs': sum(s['shortage_amount'] for s in all_shortages)
                },
                'top_shortages': all_shortages[:20]
            },
            'requirements': {
                'total_yarns_required': len(yarn_requirements),
                'total_pounds_required': sum(yarn_requirements.values()),
                'top_requirements': sorted(
                    [(k, v) for k, v in yarn_requirements.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            },
            'recommendations': cls._generate_recommendations(validation, all_shortages, yarn_requirements)
        }
        
        return report
    
    @classmethod
    def _generate_recommendations(cls,
                                 validation: Dict,
                                 shortages: List[Dict],
                                 requirements: Dict[str, float]) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Critical shortage recommendations
        critical_shortages = [s for s in shortages if s['risk_level'] == 'CRITICAL']
        if critical_shortages:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'PROCUREMENT',
                'action': f"Immediately order {len(critical_shortages)} critically short yarns",
                'impact': f"Will address {sum(s['shortage_amount'] for s in critical_shortages):.0f} lbs shortage",
                'yarns': [s['yarn_id'] for s in critical_shortages[:5]]
            })
        
        # Data consistency recommendations
        if not validation['is_consistent']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'DATA_QUALITY',
                'action': "Fix data consistency issues",
                'impact': "Ensure accurate inventory and production planning",
                'issues': validation['warnings'][:3]
            })
        
        # BOM coverage recommendations
        if validation['warnings']:
            bom_warnings = [w for w in validation['warnings'] if w['type'] == 'MISSING_BOM']
            if bom_warnings:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'MASTER_DATA',
                    'action': f"Update BOM for {len(bom_warnings[0]['styles'])} styles",
                    'impact': "Enable accurate yarn requirement calculations",
                    'styles': bom_warnings[0]['styles'][:5]
                })
        
        return recommendations