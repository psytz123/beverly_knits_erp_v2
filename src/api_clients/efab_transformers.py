"""
eFab.ai Data Transformers
Transform API responses to match Beverly Knits ERP data models
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class EFabDataTransformer:
    """
    Transform API responses to ERP data models
    Handles field mapping, data type conversion, and business logic
    """
    
    # Field mapping dictionaries for each data type
    YARN_ACTIVE_MAPPING = {
        'yarn_id': 'Desc#',
        'id': 'Desc#',
        'description': 'Yarn Description',
        'yarn_description': 'Yarn Description',
        'theoretical_balance': 'Theoretical Balance',
        'theoretical_bal': 'Theoretical Balance',
        'allocated': 'Allocated',
        'on_order': 'On Order',
        'on_order_qty': 'On Order',
        'cost_per_pound': 'Cost/lb',
        'cost_per_lb': 'Cost/lb',
        'vendor': 'Vendor',
        'vendor_name': 'Vendor',
        'lead_time_days': 'Lead Time',
        'min_order_qty': 'MOQ',
        'safety_stock': 'Safety Stock'
    }
    
    KNIT_ORDER_MAPPING = {
        'ko_number': 'KO#',
        'ko_num': 'KO#',
        'knit_order': 'KO#',
        'style': 'Style#',
        'style_num': 'Style#',
        'style_number': 'Style#',
        'qty_ordered_lbs': 'Qty Ordered (lbs)',
        'quantity_lbs': 'Qty Ordered (lbs)',
        'qty_lbs': 'Qty Ordered (lbs)',
        'machine': 'Machine',
        'machine_id': 'Machine',
        'work_center': 'Work Center',
        'wc': 'Work Center',
        'status': 'Status',
        'order_status': 'Status',
        'due_date': 'Due Date',
        'start_date': 'Start Date',
        'completion_pct': 'Completion %',
        'percent_complete': 'Completion %'
    }
    
    PO_DELIVERY_MAPPING = {
        'yarn_id': 'Desc#',
        'yarn': 'Desc#',
        'po_number': 'PO#',
        'po_num': 'PO#',
        'vendor': 'Vendor',
        'vendor_name': 'Vendor',
        'quantity': 'Quantity',
        'qty': 'Quantity',
        'delivery_date': 'Delivery Date',
        'expected_date': 'Delivery Date',
        'status': 'Status',
        'po_status': 'Status'
    }
    
    SALES_ACTIVITY_MAPPING = {
        'style': 'Style#',
        'style_num': 'Style#',
        'customer': 'Customer',
        'customer_name': 'Customer',
        'quantity': 'Quantity',
        'qty': 'Quantity',
        'unit_price': 'Unit Price',
        'price': 'Unit Price',
        'total_amount': 'Line Price',
        'line_total': 'Line Price',
        'order_date': 'Order Date',
        'ship_date': 'Ship Date',
        'status': 'Status'
    }
    
    def __init__(self):
        """Initialize transformer"""
        self.current_week = self._get_current_week_number()
    
    def _get_current_week_number(self) -> int:
        """Get current week number"""
        return datetime.now().isocalendar()[1]
    
    def _clean_numeric_field(self, value: Any) -> float:
        """
        Clean and convert numeric field
        
        Args:
            value: Raw value (string, float, etc.)
            
        Returns:
            Clean float value
        """
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert string to float
        if isinstance(value, str):
            # Remove currency symbols
            value = value.replace('$', '').replace('£', '').replace('€', '')
            # Remove commas
            value = value.replace(',', '')
            # Remove spaces
            value = value.strip()
            
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        return 0.0
    
    def _clean_date_field(self, value: Any) -> Optional[datetime]:
        """
        Clean and convert date field
        
        Args:
            value: Raw date value
            
        Returns:
            datetime object or None
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def transform_yarn_active(self, api_response: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform /api/yarn/active response to yarn inventory format
        
        Args:
            api_response: API response (dict, list, or DataFrame)
            
        Returns:
            Transformed DataFrame matching ERP format
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(api_response, dict):
                if 'data' in api_response:
                    df = pd.DataFrame(api_response['data'])
                else:
                    df = pd.DataFrame([api_response])
            elif isinstance(api_response, list):
                df = pd.DataFrame(api_response)
            else:
                df = api_response.copy()
            
            if df.empty:
                logger.warning("Empty yarn active response")
                return pd.DataFrame()
            
            # Apply field mapping
            df_mapped = pd.DataFrame()
            for api_field, erp_field in self.YARN_ACTIVE_MAPPING.items():
                if api_field in df.columns:
                    df_mapped[erp_field] = df[api_field]
            
            # Ensure required fields exist
            required_fields = ['Desc#', 'Yarn Description', 'Theoretical Balance', 'Allocated', 'On Order']
            for field in required_fields:
                if field not in df_mapped.columns:
                    logger.warning(f"Missing required field: {field}")
                    df_mapped[field] = 0 if field != 'Yarn Description' else ''
            
            # Clean numeric fields
            numeric_fields = ['Theoretical Balance', 'Allocated', 'On Order', 'Cost/lb', 'MOQ', 'Safety Stock']
            for field in numeric_fields:
                if field in df_mapped.columns:
                    df_mapped[field] = df_mapped[field].apply(self._clean_numeric_field)
            
            # Calculate Planning Balance
            df_mapped['Planning Balance'] = (
                df_mapped['Theoretical Balance'] +
                df_mapped['Allocated'] +  # Already negative in most cases
                df_mapped['On Order']
            )
            
            # Add additional calculated fields
            df_mapped['Available'] = df_mapped['Theoretical Balance'].clip(lower=0)
            df_mapped['Shortage'] = (-df_mapped['Planning Balance']).clip(lower=0)
            
            # Sort by Desc#
            if 'Desc#' in df_mapped.columns:
                df_mapped = df_mapped.sort_values('Desc#')
            
            logger.info(f"Transformed {len(df_mapped)} yarn inventory records")
            return df_mapped
            
        except Exception as e:
            logger.error(f"Error transforming yarn active data: {e}")
            return pd.DataFrame()
    
    def transform_knit_orders(self, api_response: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform knit orders response to ERP format
        
        Args:
            api_response: API response
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Convert to DataFrame
            if isinstance(api_response, dict):
                df = pd.DataFrame(api_response.get('data', []))
            elif isinstance(api_response, list):
                df = pd.DataFrame(api_response)
            else:
                df = api_response.copy()
            
            if df.empty:
                return pd.DataFrame()
            
            # Apply field mapping
            df_mapped = pd.DataFrame()
            for api_field, erp_field in self.KNIT_ORDER_MAPPING.items():
                if api_field in df.columns:
                    df_mapped[erp_field] = df[api_field]
            
            # Clean numeric fields
            if 'Qty Ordered (lbs)' in df_mapped.columns:
                df_mapped['Qty Ordered (lbs)'] = df_mapped['Qty Ordered (lbs)'].apply(self._clean_numeric_field)
            
            if 'Completion %' in df_mapped.columns:
                df_mapped['Completion %'] = df_mapped['Completion %'].apply(self._clean_numeric_field)
            
            # Clean date fields
            date_fields = ['Due Date', 'Start Date']
            for field in date_fields:
                if field in df_mapped.columns:
                    df_mapped[field] = df_mapped[field].apply(self._clean_date_field)
            
            # Determine assignment status
            if 'Machine' in df_mapped.columns:
                df_mapped['Assigned'] = df_mapped['Machine'].notna() & (df_mapped['Machine'] != '')
            
            logger.info(f"Transformed {len(df_mapped)} knit order records")
            return df_mapped
            
        except Exception as e:
            logger.error(f"Error transforming knit orders: {e}")
            return pd.DataFrame()
    
    def transform_yarn_expected(self, api_response: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform /api/report/yarn_expected for time-phased planning
        Maps deliveries to weekly buckets for PODeliveryLoader
        
        Args:
            api_response: API response with PO delivery data
            
        Returns:
            DataFrame formatted for PODeliveryLoader with weekly columns
        """
        try:
            # Handle different response formats
            if isinstance(api_response, dict):
                if 'data' in api_response:
                    data = api_response['data']
                else:
                    data = api_response
            else:
                data = api_response
            
            # Initialize result DataFrame
            result_rows = []
            
            # Process each yarn's deliveries
            if isinstance(data, dict):
                # Format: {yarn_id: {deliveries: {...}}}
                for yarn_id, yarn_data in data.items():
                    row = {'yarn_id': yarn_id}
                    
                    if isinstance(yarn_data, dict) and 'deliveries' in yarn_data:
                        deliveries = yarn_data['deliveries']
                    else:
                        deliveries = yarn_data
                    
                    # Process delivery buckets
                    row = self._process_delivery_buckets(row, deliveries)
                    result_rows.append(row)
                    
            elif isinstance(data, list):
                # Format: [{yarn_id: ..., deliveries: {...}}, ...]
                for item in data:
                    if isinstance(item, dict):
                        row = {'yarn_id': item.get('yarn_id', item.get('yarn', ''))}
                        deliveries = item.get('deliveries', {})
                        row = self._process_delivery_buckets(row, deliveries)
                        result_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(result_rows)
            
            if df.empty:
                # Return empty DataFrame with expected columns
                return self._create_empty_po_dataframe()
            
            # Ensure all week columns exist
            week_columns = ['week_past_due'] + [f'week_{i}' for i in range(36, 45)] + ['week_later']
            for col in week_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Clean numeric values
            for col in week_columns:
                df[col] = df[col].fillna(0).astype(float)
            
            # Add metadata columns
            df['Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Transformed {len(df)} yarn expected delivery records")
            return df
            
        except Exception as e:
            logger.error(f"Error transforming yarn expected data: {e}")
            return self._create_empty_po_dataframe()
    
    def _process_delivery_buckets(self, row: Dict, deliveries: Dict) -> Dict:
        """
        Process delivery data into weekly buckets
        
        Args:
            row: Row dictionary to populate
            deliveries: Delivery data dictionary
            
        Returns:
            Updated row with weekly buckets
        """
        current_date = datetime.now()
        current_week = current_date.isocalendar()[1]
        
        # Process each delivery entry
        for date_key, quantity in deliveries.items():
            qty = self._clean_numeric_field(quantity)
            
            if date_key in ['past_due', 'past', 'overdue']:
                row['week_past_due'] = row.get('week_past_due', 0) + qty
                
            elif date_key in ['later', 'future', 'beyond']:
                row['week_later'] = row.get('week_later', 0) + qty
                
            else:
                # Try to parse as date
                delivery_date = self._clean_date_field(date_key)
                if delivery_date:
                    # Calculate week number
                    delivery_week = delivery_date.isocalendar()[1]
                    
                    # Map to week column
                    if delivery_date < current_date:
                        row['week_past_due'] = row.get('week_past_due', 0) + qty
                    elif 36 <= delivery_week <= 44:
                        col = f'week_{delivery_week}'
                        row[col] = row.get(col, 0) + qty
                    else:
                        row['week_later'] = row.get('week_later', 0) + qty
                else:
                    # Check if it's a week column directly
                    week_match = re.match(r'week[_\s]?(\d+)', date_key, re.IGNORECASE)
                    if week_match:
                        week_num = int(week_match.group(1))
                        if 36 <= week_num <= 44:
                            col = f'week_{week_num}'
                            row[col] = row.get(col, 0) + qty
        
        return row
    
    def _create_empty_po_dataframe(self) -> pd.DataFrame:
        """Create empty PO DataFrame with expected columns"""
        columns = ['yarn_id', 'week_past_due'] + \
                  [f'week_{i}' for i in range(36, 45)] + \
                  ['week_later', 'Last Updated']
        
        df = pd.DataFrame(columns=columns)
        # Set numeric columns to float type
        for col in columns:
            if col.startswith('week'):
                df[col] = df[col].astype(float)
        
        return df
    
    def transform_sales_activity(self, api_response: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform sales activity response to ERP format
        
        Args:
            api_response: API response
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Convert to DataFrame
            if isinstance(api_response, dict):
                df = pd.DataFrame(api_response.get('data', []))
            elif isinstance(api_response, list):
                df = pd.DataFrame(api_response)
            else:
                df = api_response.copy()
            
            if df.empty:
                return pd.DataFrame()
            
            # Apply field mapping
            df_mapped = pd.DataFrame()
            for api_field, erp_field in self.SALES_ACTIVITY_MAPPING.items():
                if api_field in df.columns:
                    df_mapped[erp_field] = df[api_field]
            
            # Clean price fields (remove $ symbols)
            price_fields = ['Unit Price', 'Line Price']
            for field in price_fields:
                if field in df_mapped.columns:
                    df_mapped[field] = df_mapped[field].apply(self._clean_numeric_field)
            
            # Clean quantity
            if 'Quantity' in df_mapped.columns:
                df_mapped['Quantity'] = df_mapped['Quantity'].apply(self._clean_numeric_field)
            
            # Clean dates
            date_fields = ['Order Date', 'Ship Date']
            for field in date_fields:
                if field in df_mapped.columns:
                    df_mapped[field] = df_mapped[field].apply(self._clean_date_field)
            
            # Calculate line total if missing
            if 'Line Price' not in df_mapped.columns and all(f in df_mapped.columns for f in ['Unit Price', 'Quantity']):
                df_mapped['Line Price'] = df_mapped['Unit Price'] * df_mapped['Quantity']
            
            logger.info(f"Transformed {len(df_mapped)} sales activity records")
            return df_mapped
            
        except Exception as e:
            logger.error(f"Error transforming sales activity: {e}")
            return pd.DataFrame()
    
    def transform_greige_inventory(self, api_response: Union[Dict, List, pd.DataFrame], stage: str = 'g00') -> pd.DataFrame:
        """
        Transform greige inventory response
        
        Args:
            api_response: API response
            stage: Greige stage (g00 or g02)
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Convert to DataFrame
            if isinstance(api_response, dict):
                df = pd.DataFrame(api_response.get('data', []))
            elif isinstance(api_response, list):
                df = pd.DataFrame(api_response)
            else:
                df = api_response.copy()
            
            if df.empty:
                return pd.DataFrame()
            
            # Standard field mapping
            field_map = {
                'style': 'Style#',
                'style_num': 'Style#',
                'quantity': 'Quantity',
                'qty': 'Quantity',
                'weight_lbs': 'Weight (lbs)',
                'weight': 'Weight (lbs)',
                'location': 'Location',
                'lot_number': 'Lot#',
                'lot': 'Lot#'
            }
            
            df_mapped = pd.DataFrame()
            for api_field, erp_field in field_map.items():
                if api_field in df.columns:
                    df_mapped[erp_field] = df[api_field]
            
            # Add stage column
            df_mapped['Stage'] = stage.upper()
            
            # Clean numeric fields
            numeric_fields = ['Quantity', 'Weight (lbs)']
            for field in numeric_fields:
                if field in df_mapped.columns:
                    df_mapped[field] = df_mapped[field].apply(self._clean_numeric_field)
            
            logger.info(f"Transformed {len(df_mapped)} greige {stage} records")
            return df_mapped
            
        except Exception as e:
            logger.error(f"Error transforming greige inventory: {e}")
            return pd.DataFrame()
    
    def validate_transformation(self, original: pd.DataFrame, transformed: pd.DataFrame, data_type: str) -> bool:
        """
        Validate transformation results
        
        Args:
            original: Original DataFrame
            transformed: Transformed DataFrame
            data_type: Type of data being validated
            
        Returns:
            True if validation passes
        """
        try:
            # Check if transformation produced data
            if transformed.empty:
                logger.warning(f"Transformation produced empty DataFrame for {data_type}")
                return False
            
            # Check required fields based on data type
            required_fields = {
                'yarn_inventory': ['Desc#', 'Planning Balance'],
                'knit_orders': ['KO#', 'Style#', 'Qty Ordered (lbs)'],
                'po_deliveries': ['yarn_id', 'week_past_due'],
                'sales_activity': ['Style#', 'Quantity']
            }
            
            if data_type in required_fields:
                missing = [f for f in required_fields[data_type] if f not in transformed.columns]
                if missing:
                    logger.error(f"Missing required fields for {data_type}: {missing}")
                    return False
            
            # Check for data loss
            if len(original) > 0:
                retention_rate = len(transformed) / len(original)
                if retention_rate < 0.9:
                    logger.warning(f"Low data retention rate for {data_type}: {retention_rate:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {data_type}: {e}")
            return False


def test_transformer():
    """Test data transformer functionality"""
    transformer = EFabDataTransformer()
    
    # Test yarn active transformation
    print("Testing Yarn Active Transformation:")
    yarn_data = {
        'data': [
            {
                'yarn_id': '18884',
                'description': '100% COTTON 30/1 ROYAL BLUE',
                'theoretical_balance': '2,506.18',
                'allocated': '-30,859.80',
                'on_order': '36,161.30'
            }
        ]
    }
    
    yarn_df = transformer.transform_yarn_active(yarn_data)
    print(f"  Columns: {list(yarn_df.columns)}")
    print(f"  Planning Balance: {yarn_df['Planning Balance'].iloc[0] if not yarn_df.empty else 'N/A'}")
    
    # Test yarn expected transformation
    print("\nTesting Yarn Expected Transformation:")
    po_data = {
        'data': {
            '18884': {
                'deliveries': {
                    'past_due': 20161.30,
                    '2025-10-10': 4000,
                    '2025-10-17': 4000,
                    'later': 8000
                }
            }
        }
    }
    
    po_df = transformer.transform_yarn_expected(po_data)
    print(f"  Columns: {list(po_df.columns)}")
    print(f"  Past due: {po_df['week_past_due'].iloc[0] if not po_df.empty else 'N/A'}")
    
    # Test validation
    print("\nTesting Validation:")
    is_valid = transformer.validate_transformation(
        pd.DataFrame(yarn_data['data']),
        yarn_df,
        'yarn_inventory'
    )
    print(f"  Validation result: {is_valid}")


if __name__ == "__main__":
    test_transformer()