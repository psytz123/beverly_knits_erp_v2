#!/usr/bin/env python3
"""
PO Delivery Loader Module
Loads and processes time-phased Purchase Order delivery schedules
from Expected_Yarn_Report.xlsx with weekly time buckets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import re

# Configure logging
logger = logging.getLogger(__name__)


class PODeliveryLoader:
    """
    Loads and processes PO delivery schedules with time buckets
    Converts Expected_Yarn_Report.xlsx into time-phased delivery data
    """
    
    def __init__(self):
        """Initialize PO Delivery Loader with expected column mapping"""
        
        # Standard delivery columns as they appear in Expected_Yarn_Report
        self.delivery_columns = [
            'Unscheduled or Past Due',
            'This Week',
            '9/5/2025',
            '9/12/2025', 
            '9/19/2025',
            '9/26/2025',
            '10/3/2025',
            '10/10/2025',
            '10/17/2025',
            '10/24/2025',
            'Later'
        ]
        
        # Column aliases for flexible parsing
        self.column_aliases = {
            'Desc': ['Desc', 'desc', 'Yarn_ID', 'YarnID'],
            'PO Bal': ['PO Bal', 'PO_Bal', 'Balance'],
            'Unscheduled or Past Due': ['Unscheduled or Past Due', 'Past Due', 'Unscheduled'],
            'This Week': ['This Week', 'Current Week'],
            'Later': ['Later', 'Future']
        }
        
        # Week number mapping for time-phased calculations
        self.week_mapping = {}
        self._initialize_week_mapping()
    
    def _initialize_week_mapping(self):
        """Initialize date to week number mapping"""
        # Map delivery dates to sequential week numbers
        base_date = datetime(2025, 9, 5)  # Week 36 starts here
        
        date_columns = [
            '9/5/2025', '9/12/2025', '9/19/2025', '9/26/2025',
            '10/3/2025', '10/10/2025', '10/17/2025', '10/24/2025'
        ]
        
        for i, date_str in enumerate(date_columns):
            week_num = 36 + i  # Start at week 36
            self.week_mapping[date_str] = week_num
            self.week_mapping[f'week_{week_num}'] = date_str
    
    def load_po_deliveries(self, file_path: str) -> pd.DataFrame:
        """
        Load Expected_Yarn_Report.xlsx with delivery timing
        
        Args:
            file_path: Path to Expected_Yarn_Report.xlsx or .csv
            
        Returns:
            DataFrame with PO delivery data
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded PO delivery data: {len(df)} rows from {file_path}")
            
            # Clean and standardize column names
            df = self._clean_column_names(df)
            
            # Clean numeric fields (remove commas, dollar signs)
            df = self._clean_numeric_fields(df)
            
            # Validate required columns exist
            self._validate_delivery_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading PO deliveries from {file_path}: {e}")
            raise
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        # Remove BOM character if present
        if df.columns[0].startswith('\ufeff'):
            df.columns = [df.columns[0].replace('\ufeff', '')] + list(df.columns[1:])
        
        return df
    
    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric fields by removing commas and converting to float"""
        numeric_columns = self.delivery_columns + ['PO Bal', 'Price']
        
        for col in numeric_columns:
            if col in df.columns:
                # Handle string values with commas and quotes
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(',', '').str.replace('"', '').str.replace('$', '')
                
                # Convert to numeric, handling empty strings and NaN
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _validate_delivery_columns(self, df: pd.DataFrame):
        """Validate that required delivery columns exist"""
        missing_columns = []
        
        for col in self.delivery_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            available_cols = list(df.columns)
            logger.warning(f"Missing delivery columns: {missing_columns}")
            logger.info(f"Available columns: {available_cols}")
            
            # Don't raise error, just log - some columns might be optional
    
    def map_to_weekly_buckets(self, po_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to standardized week numbers
        
        Args:
            po_data: DataFrame with delivery date columns
            
        Returns:
            DataFrame with week_XX columns added
        """
        df = po_data.copy()
        
        # Add week number columns
        for date_col, week_num in self.week_mapping.items():
            if isinstance(week_num, int) and date_col in df.columns:
                week_col = f'week_{week_num}'
                df[week_col] = df[date_col]
        
        # Add special handling for past due and later columns
        if 'Unscheduled or Past Due' in df.columns:
            df['week_past_due'] = df['Unscheduled or Past Due']
        
        if 'This Week' in df.columns:
            df['week_current'] = df['This Week']
            
        if 'Later' in df.columns:
            df['week_later'] = df['Later']
        
        logger.info("Mapped delivery dates to weekly buckets")
        return df
    
    def aggregate_by_yarn(self, po_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Group PO deliveries by yarn with weekly totals
        
        Args:
            po_data: DataFrame with weekly bucket columns
            
        Returns:
            Dictionary mapping yarn_id to weekly delivery amounts
        """
        yarn_deliveries = {}
        
        # Identify yarn ID column
        yarn_col = None
        for col_name, aliases in self.column_aliases.items():
            if col_name == 'Desc':
                for alias in aliases:
                    if alias in po_data.columns:
                        yarn_col = alias
                        break
                break
        
        if not yarn_col:
            logger.error("No yarn ID column found in data")
            return {}
        
        # Group by yarn and sum weekly deliveries
        for _, row in po_data.iterrows():
            yarn_id = str(row[yarn_col])
            
            if yarn_id not in yarn_deliveries:
                yarn_deliveries[yarn_id] = {}
            
            # Sum deliveries for each time bucket
            weekly_totals = {}
            
            # Past due/current week
            if 'week_past_due' in po_data.columns:
                weekly_totals['past_due'] = row.get('week_past_due', 0)
            if 'week_current' in po_data.columns:
                weekly_totals['current'] = row.get('week_current', 0)
            
            # Future weeks (36-44)
            for week_num in range(36, 45):
                week_col = f'week_{week_num}'
                if week_col in po_data.columns:
                    weekly_totals[f'week_{week_num}'] = row.get(week_col, 0)
            
            # Later deliveries
            if 'week_later' in po_data.columns:
                weekly_totals['later'] = row.get('week_later', 0)
            
            # Aggregate with existing entries for same yarn
            for week, amount in weekly_totals.items():
                if week not in yarn_deliveries[yarn_id]:
                    yarn_deliveries[yarn_id][week] = 0
                yarn_deliveries[yarn_id][week] += amount
        
        logger.info(f"Aggregated deliveries for {len(yarn_deliveries)} yarns")
        return yarn_deliveries
    
    def calculate_total_on_order(self, weekly_deliveries: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate total on order amounts for each yarn
        
        Args:
            weekly_deliveries: Yarn delivery schedules by week
            
        Returns:
            Dictionary mapping yarn_id to total on order amount
        """
        totals = {}
        
        for yarn_id, weekly_amounts in weekly_deliveries.items():
            total = sum(weekly_amounts.values())
            totals[yarn_id] = total
        
        return totals
    
    def get_delivery_timeline(self, yarn_id: str, weekly_deliveries: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Get delivery timeline for specific yarn
        
        Args:
            yarn_id: Yarn identifier
            weekly_deliveries: Complete delivery schedule data
            
        Returns:
            Weekly delivery amounts for the specified yarn
        """
        return weekly_deliveries.get(str(yarn_id), {})
    
    def get_next_receipt_week(self, yarn_id: str, weekly_deliveries: Dict[str, Dict[str, float]]) -> Optional[str]:
        """
        Find the next week with scheduled receipts for a yarn
        
        Args:
            yarn_id: Yarn identifier  
            weekly_deliveries: Complete delivery schedule data
            
        Returns:
            Next week with receipts, or None if no future receipts
        """
        yarn_schedule = self.get_delivery_timeline(yarn_id, weekly_deliveries)
        
        # Check weeks in chronological order
        for week_num in range(36, 45):
            week_key = f'week_{week_num}'
            if yarn_schedule.get(week_key, 0) > 0:
                return week_key
        
        # Check later bucket
        if yarn_schedule.get('later', 0) > 0:
            return 'later'
            
        return None
    
    def export_time_phased_data(self, weekly_deliveries: Dict[str, Dict[str, float]], 
                              output_path: str) -> bool:
        """
        Export time-phased delivery data to CSV for validation
        
        Args:
            weekly_deliveries: Yarn delivery schedules
            output_path: Output file path
            
        Returns:
            Success status
        """
        try:
            # Convert to DataFrame for export
            export_data = []
            
            for yarn_id, weekly_amounts in weekly_deliveries.items():
                row = {'yarn_id': yarn_id}
                row.update(weekly_amounts)
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported time-phased data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False


def main():
    """Test function for PO Delivery Loader"""
    
    # Initialize loader
    loader = PODeliveryLoader()
    
    # Test file path
    test_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.csv"
    
    try:
        # Load PO delivery data
        po_data = loader.load_po_deliveries(test_file)
        print(f"Loaded {len(po_data)} PO records")
        
        # Map to weekly buckets
        weekly_data = loader.map_to_weekly_buckets(po_data)
        print("Mapped to weekly buckets")
        
        # Aggregate by yarn
        yarn_deliveries = loader.aggregate_by_yarn(weekly_data)
        print(f"Aggregated deliveries for {len(yarn_deliveries)} yarns")
        
        # Calculate totals
        totals = loader.calculate_total_on_order(yarn_deliveries)
        
        # Show sample results
        sample_yarns = list(yarn_deliveries.keys())[:5]
        for yarn_id in sample_yarns:
            timeline = loader.get_delivery_timeline(yarn_id, yarn_deliveries)
            total = totals.get(yarn_id, 0)
            next_receipt = loader.get_next_receipt_week(yarn_id, yarn_deliveries)
            
            print(f"\nYarn {yarn_id}:")
            print(f"  Total On Order: {total:,.2f}")
            print(f"  Next Receipt: {next_receipt}")
            print(f"  Timeline: {timeline}")
        
        # Export test data
        output_path = "/tmp/time_phased_deliveries_test.csv"
        loader.export_time_phased_data(yarn_deliveries, output_path)
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()