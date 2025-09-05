"""Data Validator - Ensure data quality and consistency."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging


class DataValidator:
    """
    Validate data quality and consistency for Beverly Knits ERP.
    Performs both structural and business rule validations.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger(__name__)
        
        # Define validation rules by data type
        self.validation_rules = {
            'yarn_inventory': {
                'required_columns': ['yarn_id', 'description', 'theoretical_balance'],
                'numeric_columns': ['theoretical_balance', 'allocated', 'on_order'],
                'non_negative': ['theoretical_balance', 'on_order'],
                'unique_key': 'yarn_id',
                'business_rules': [
                    self._validate_yarn_balances,
                    self._validate_yarn_ids
                ]
            },
            'bom_data': {
                'required_columns': ['style_id', 'yarn_id', 'quantity'],
                'numeric_columns': ['quantity', 'bom_percent'],
                'non_negative': ['quantity', 'bom_percent'],
                'composite_key': ['style_id', 'yarn_id'],
                'business_rules': [
                    self._validate_bom_percentages,
                    self._validate_bom_quantities
                ]
            },
            'production_orders': {
                'required_columns': ['order_id', 'style_id', 'qty_ordered'],
                'numeric_columns': ['qty_ordered', 'qty_produced'],
                'non_negative': ['qty_ordered', 'qty_produced'],
                'unique_key': 'order_id',
                'date_columns': ['due_date', 'scheduled_date'],
                'business_rules': [
                    self._validate_production_quantities,
                    self._validate_order_dates
                ]
            },
            'work_centers': {
                'required_columns': ['work_center', 'machine_id'],
                'unique_key': 'machine_id',
                'business_rules': [
                    self._validate_work_center_patterns
                ]
            },
            'sales_activity': {
                'required_columns': ['date', 'style_id', 'quantity', 'price'],
                'numeric_columns': ['quantity', 'price', 'total_cost'],
                'non_negative': ['quantity', 'price'],
                'date_columns': ['date', 'ship_date'],
                'business_rules': [
                    self._validate_sales_data
                ]
            }
        }
    
    def validate(self, df: pd.DataFrame, data_type: str) -> 'ValidationResult':
        """
        Validate a DataFrame according to its data type.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        if df is None or df.empty:
            result.add_error("DataFrame is empty or None")
            return result
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, {})
        
        if not rules:
            result.add_warning(f"No validation rules defined for data type: {data_type}")
            return result
        
        # Structural validations
        self._validate_structure(df, rules, result)
        
        # Data type validations
        self._validate_data_types(df, rules, result)
        
        # Uniqueness validations
        self._validate_uniqueness(df, rules, result)
        
        # Business rule validations
        if 'business_rules' in rules:
            for rule_func in rules['business_rules']:
                try:
                    rule_func(df, result)
                except Exception as e:
                    result.add_warning(f"Business rule validation failed: {e}")
        
        # Data quality checks
        self._validate_data_quality(df, result)
        
        return result
    
    def _validate_structure(self, df: pd.DataFrame, rules: Dict, result: ValidationResult):
        """Validate DataFrame structure."""
        # Check required columns
        if 'required_columns' in rules:
            for col in rules['required_columns']:
                if col not in df.columns:
                    result.add_error(f"Missing required column: {col}")
        
        # Check for unexpected nulls in required columns
        for col in rules.get('required_columns', []):
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    result.add_warning(f"Column '{col}' has {null_count} null values")
    
    def _validate_data_types(self, df: pd.DataFrame, rules: Dict, result: ValidationResult):
        """Validate data types."""
        # Numeric columns
        if 'numeric_columns' in rules:
            for col in rules['numeric_columns']:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            # Try to convert
                            pd.to_numeric(df[col], errors='coerce')
                            result.add_warning(f"Column '{col}' should be numeric but isn't")
                        except:
                            result.add_error(f"Column '{col}' cannot be converted to numeric")
        
        # Non-negative columns
        if 'non_negative' in rules:
            for col in rules['non_negative']:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        # Exception for 'allocated' which should be negative
                        if col != 'allocated':
                            result.add_warning(f"Column '{col}' has {negative_count} negative values")
        
        # Date columns
        if 'date_columns' in rules:
            for col in rules['date_columns']:
                if col in df.columns:
                    try:
                        pd.to_datetime(df[col], errors='coerce')
                    except:
                        result.add_error(f"Column '{col}' cannot be parsed as date")
    
    def _validate_uniqueness(self, df: pd.DataFrame, rules: Dict, result: ValidationResult):
        """Validate uniqueness constraints."""
        # Single unique key
        if 'unique_key' in rules:
            key = rules['unique_key']
            if key in df.columns:
                duplicates = df[key].duplicated().sum()
                if duplicates > 0:
                    result.add_error(f"Column '{key}' has {duplicates} duplicate values")
        
        # Composite unique key
        if 'composite_key' in rules:
            keys = rules['composite_key']
            if all(k in df.columns for k in keys):
                duplicates = df.duplicated(subset=keys).sum()
                if duplicates > 0:
                    result.add_warning(f"Composite key {keys} has {duplicates} duplicate combinations")
    
    def _validate_data_quality(self, df: pd.DataFrame, result: ValidationResult):
        """Perform general data quality checks."""
        # Check for high percentage of nulls
        null_percentages = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentages[null_percentages > 50]
        
        for col, pct in high_null_cols.items():
            result.add_warning(f"Column '{col}' is {pct:.1f}% null")
        
        # Check for suspicious patterns
        for col in df.select_dtypes(include=['object']).columns:
            # Check for leading/trailing spaces
            if df[col].notna().any():
                space_issues = df[col].str.strip() != df[col]
                if space_issues.any():
                    result.add_warning(f"Column '{col}' has leading/trailing spaces")
        
        # Check for outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df[col].dropna()) > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    result.add_info(f"Column '{col}' has {outliers} potential outliers")
    
    # Business rule validation functions
    def _validate_yarn_balances(self, df: pd.DataFrame, result: ValidationResult):
        """Validate yarn inventory balance calculations."""
        if all(col in df.columns for col in ['theoretical_balance', 'allocated', 'on_order']):
            # Check if planning balance is correctly calculated
            if 'planning_balance' in df.columns:
                calculated = df['theoretical_balance'] + df['allocated'] + df['on_order']
                discrepancies = ~np.isclose(calculated, df['planning_balance'], rtol=0.01, equal_nan=True)
                
                if discrepancies.any():
                    count = discrepancies.sum()
                    result.add_warning(f"{count} rows have planning balance calculation discrepancies")
            
            # Check for critical shortages
            if 'planning_balance' in df.columns:
                critical = df['planning_balance'] < -1000
                if critical.any():
                    count = critical.sum()
                    result.add_warning(f"{count} yarns have critical negative balances (< -1000)")
    
    def _validate_yarn_ids(self, df: pd.DataFrame, result: ValidationResult):
        """Validate yarn ID format."""
        if 'yarn_id' in df.columns:
            # Check for empty or null yarn IDs
            empty = df['yarn_id'].isna() | (df['yarn_id'] == '')
            if empty.any():
                result.add_error(f"{empty.sum()} rows have empty yarn IDs")
    
    def _validate_bom_percentages(self, df: pd.DataFrame, result: ValidationResult):
        """Validate BOM percentage totals."""
        if 'bom_percent' in df.columns and 'style_id' in df.columns:
            # Check if percentages sum to ~100 for each style
            style_totals = df.groupby('style_id')['bom_percent'].sum()
            invalid_totals = ~style_totals.between(95, 105)  # Allow 5% tolerance
            
            if invalid_totals.any():
                count = invalid_totals.sum()
                result.add_warning(f"{count} styles have BOM percentages not summing to ~100%")
    
    def _validate_bom_quantities(self, df: pd.DataFrame, result: ValidationResult):
        """Validate BOM quantities."""
        if 'quantity' in df.columns:
            # Check for zero quantities
            zero_qty = df['quantity'] == 0
            if zero_qty.any():
                result.add_warning(f"{zero_qty.sum()} BOM entries have zero quantity")
    
    def _validate_production_quantities(self, df: pd.DataFrame, result: ValidationResult):
        """Validate production order quantities."""
        if 'qty_ordered' in df.columns and 'qty_produced' in df.columns:
            # Check for over-production
            over_produced = df['qty_produced'] > df['qty_ordered']
            if over_produced.any():
                count = over_produced.sum()
                result.add_warning(f"{count} orders have produced more than ordered")
    
    def _validate_order_dates(self, df: pd.DataFrame, result: ValidationResult):
        """Validate production order dates."""
        if 'due_date' in df.columns:
            # Check for past due dates with incomplete orders
            df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
            today = pd.Timestamp.now()
            
            if 'status' in df.columns:
                past_due = (df['due_date'] < today) & (df['status'] != 'completed')
                if past_due.any():
                    count = past_due.sum()
                    result.add_warning(f"{count} orders are past due")
    
    def _validate_work_center_patterns(self, df: pd.DataFrame, result: ValidationResult):
        """Validate work center naming patterns."""
        if 'work_center' in df.columns:
            # Check for expected pattern: x.xx.xx.X
            pattern = r'^\d\.\d{2}\.\d{2}\.[A-Z]$'
            invalid = ~df['work_center'].str.match(pattern, na=False)
            
            if invalid.any():
                count = invalid.sum()
                result.add_info(f"{count} work centers don't match standard pattern")
    
    def _validate_sales_data(self, df: pd.DataFrame, result: ValidationResult):
        """Validate sales activity data."""
        if 'price' in df.columns and 'quantity' in df.columns and 'total_cost' in df.columns:
            # Check if total = price * quantity
            calculated_total = df['price'] * df['quantity']
            discrepancies = ~np.isclose(calculated_total, df['total_cost'], rtol=0.01, equal_nan=True)
            
            if discrepancies.any():
                count = discrepancies.sum()
                result.add_warning(f"{count} sales records have total cost calculation discrepancies")


class ValidationResult:
    """Result of data validation."""
    
    def __init__(self):
        """Initialize validation result."""
        self.errors = []
        self.warnings = []
        self.info = []
        self.timestamp = datetime.now()
    
    def add_error(self, message: str):
        """Add an error (critical issue)."""
        self.errors.append(message)
    
    def add_warning(self, message: str):
        """Add a warning (potential issue)."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add info (non-critical observation)."""
        self.info.append(message)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.errors:
            parts.append(f"Errors({len(self.errors)}): {'; '.join(self.errors[:3])}")
        if self.warnings:
            parts.append(f"Warnings({len(self.warnings)}): {'; '.join(self.warnings[:3])}")
        if not parts:
            parts.append("Valid")
        return " | ".join(parts)