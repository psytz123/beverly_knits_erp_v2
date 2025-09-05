"""Column Mapper - Handle all column name variations across data sources."""

import pandas as pd
from typing import Dict, List, Set, Optional
import logging


class ColumnMapper:
    """
    Handle all column name variations in the Beverly Knits system.
    Standardizes column names across different data sources.
    """
    
    def __init__(self):
        """Initialize column mapper with master mappings."""
        self.logger = logging.getLogger(__name__)
        
        # Master mapping configuration - maps standard names to all variations
        self.MAPPINGS = {
            # Yarn identifiers
            'yarn_id': ['Desc#', 'desc_num', 'YarnID', 'yarn_id', 'Yarn ID', 'Material_ID', 'Yarn', 'yarn'],
            
            # Planning and balances
            'planning_balance': ['Planning Balance', 'Planning_Balance', 'planning balance', 'Planning_Ballance'],  # Note typo
            'theoretical_balance': ['Theoretical Balance', 'theoretical_balance', 'Theo Balance', 'Beginning Balance'],
            'allocated': ['Allocated', 'allocated', 'Alloc', 'Reserved'],
            'on_order': ['On Order', 'on_order', 'OnOrder', 'On_Order', 'Ordered'],
            
            # Style identifiers
            'style_id': ['fStyle#', 'Style#', 'style_num', 'Style', 'style_id', 'Style_ID'],
            'g_style': ['gStyle', 'gStyle#', 'g_style'],
            
            # Quantities
            'quantity': ['Qty', 'Quantity', 'quantity', 'Amount', 'Qty (lbs)', 'Qty (yds)'],
            'qty_ordered': ['Qty Ordered (lbs)', 'qty_ordered_lbs', 'Ordered (lbs)', 'Qty Ordered'],
            'qty_produced': ['Qty Produced', 'qty_produced', 'Produced (lbs)', 'Qty Made'],
            'balance': ['Balance (lbs)', 'Balance', 'balance_lbs', 'Remaining'],
            
            # Descriptions
            'description': ['Description', 'Desc', 'description', 'desc', 'Material_Name', 'Item Description'],
            
            # Dates
            'ship_date': ['Ship Date', 'ship_date', 'Shipping_Date', 'Ship_Date'],
            'due_date': ['Due Date', 'due_date', 'Due_Date', 'Required Date'],
            'scheduled_date': ['Scheduled Date', 'scheduled_date', 'Schedule_Date', 'Knit Date'],
            'created_at': ['Created At', 'created_at', 'Creation_Date', 'Date Created'],
            'updated_at': ['Updated At', 'updated_at', 'Last_Modified', 'Modified Date'],
            
            # Order identifiers
            'order_id': ['Order #', 'Order#', 'order_number', 'Order_Number', 'OrderID'],
            'po_number': ['PO#', 'PO #', 'po_number', 'PO_Number', 'Purchase_Order'],
            'ko_number': ['KO #', 'KO#', 'Actions', 'Knit_Order'],  # Knit Order
            'so_number': ['SO #', 'SO#', 'so_number', 'Sales_Order'],
            
            # Customer/Supplier
            'customer': ['Customer', 'customer', 'Client', 'client', 'Customer_Name', 'Sold To'],
            'supplier': ['Supplier', 'supplier', 'Vendor', 'vendor', 'Supplier_Name', 'Purchased From'],
            
            # Cost/Price
            'unit_price': ['Unit Price', 'unit_price', 'Price', 'Unit_Price', 'Price Per Unit'],
            'total_cost': ['Total Cost', 'Total_Cost', 'Total_Cast', 'total_cost', 'Total_Value'],  # Note typo
            'cost_per_pound': ['Cost/Pound', 'Cost_Pound', 'cost_per_pound', 'Unit_Cost', 'Price_Per_Pound'],
            
            # Status
            'status': ['Status', 'status', 'Order_Status', 'State'],
            
            # Machine/Equipment
            'machine_id': ['Machine', 'machine', 'Equipment', 'equipment', 'Machine_ID', 'MachineID'],
            'work_center': ['Work Center', 'work_center', 'WC', 'Work_Center', 'WorkCenter'],
            
            # BOM specific
            'bom_percent': ['BOM_Percent', 'BOM_Percentage', 'Percentage', 'BOM%', 'bom_percentage', 'Usage_Percentage'],
            
            # Color
            'color': ['Color', 'color', 'Colour', 'colour', 'Color_Name'],
            
            # Location
            'rack': ['Rack', 'rack', 'Location', 'location', 'Warehouse_Location', 'Bin'],
            
            # Inventory movements
            'received': ['Received', 'received', 'Receipts', 'Qty_Received', 'Received Qty'],
            'consumed': ['Consumed', 'consumed', 'Usage', 'Qty_Used', 'Used'],
            'adjustments': ['Adjustments', 'adjustments', 'Adj', 'Inventory_Adj', 'Adjustment'],
            
            # Roll information
            'roll_number': ['Roll #', 'Roll#', 'roll_number', 'Roll_Number', 'Roll'],
            'vendor_roll': ['Vendor Roll #', 'vendor_roll_number', 'Supplier_Roll', 'Vendor Roll'],
            
            # Lead time
            'lead_time': ['Lead Time', 'lead_time', 'Lead_Time', 'Lead Time Days', 'LeadTime'],
            
            # Min/Max levels
            'min_stock': ['Min Stock', 'min_stock', 'Min_Stock', 'Minimum Stock', 'Min Level'],
            'max_stock': ['Max Stock', 'max_stock', 'Max_Stock', 'Maximum Stock', 'Max Level'],
            
            # Unit of measure
            'uom': ['UOM', 'uom', 'Unit', 'unit', 'Unit_of_Measure', 'Units'],
            
            # Demand fields
            'total_demand': ['Total Demand', 'total_demand', 'Demand', 'Total_Demand'],
            'total_receipt': ['Total Receipt', 'total_receipt', 'Expected_Receipt', 'Receipts']
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_map = {}
        for standard, variations in self.MAPPINGS.items():
            for var in variations:
                self.reverse_map[var] = standard
                # Also add lowercase version
                self.reverse_map[var.lower()] = standard
    
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize all column names in a DataFrame.
        
        Args:
            df: Input DataFrame with varying column names
            
        Returns:
            DataFrame with standardized column names
        """
        if df is None or df.empty:
            return df
        
        standardized = df.copy()
        
        # Track what we're renaming
        renames = {}
        unmapped = []
        
        for col in df.columns:
            # Check if this column has a standard mapping
            standard_name = self.reverse_map.get(col)
            
            if not standard_name:
                # Try lowercase
                standard_name = self.reverse_map.get(col.lower())
            
            if standard_name:
                if col != standard_name:
                    renames[col] = standard_name
            else:
                unmapped.append(col)
        
        # Apply renames
        if renames:
            standardized = standardized.rename(columns=renames)
            self.logger.debug(f"Renamed columns: {renames}")
        
        # Log unmapped columns
        if unmapped:
            self.logger.warning(f"Unmapped columns (kept as-is): {unmapped}")
        
        return standardized
    
    def validate_required_columns(
        self, 
        df: pd.DataFrame, 
        required: List[str]
    ) -> ValidationResult:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required: List of required standard column names
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        for req_col in required:
            if req_col not in df.columns:
                # Check if any variation exists
                variations = self.MAPPINGS.get(req_col, [])
                found = False
                
                for var in variations:
                    if var in df.columns:
                        found = True
                        result.add_warning(
                            f"Required column '{req_col}' found as '{var}' - consider standardizing"
                        )
                        break
                
                if not found:
                    result.add_error(f"Missing required column: {req_col}")
        
        return result
    
    def get_standard_name(self, column: str) -> Optional[str]:
        """
        Get the standard name for a column variation.
        
        Args:
            column: Column name to standardize
            
        Returns:
            Standard column name or None if not found
        """
        return self.reverse_map.get(column) or self.reverse_map.get(column.lower())
    
    def get_variations(self, standard_name: str) -> List[str]:
        """
        Get all known variations of a standard column name.
        
        Args:
            standard_name: Standard column name
            
        Returns:
            List of all known variations
        """
        return self.MAPPINGS.get(standard_name, [])
    
    def add_mapping(self, standard_name: str, variation: str):
        """
        Add a new column variation mapping.
        
        Args:
            standard_name: Standard column name
            variation: New variation to map
        """
        if standard_name not in self.MAPPINGS:
            self.MAPPINGS[standard_name] = []
        
        if variation not in self.MAPPINGS[standard_name]:
            self.MAPPINGS[standard_name].append(variation)
            self.reverse_map[variation] = standard_name
            self.reverse_map[variation.lower()] = standard_name
            self.logger.info(f"Added mapping: {variation} -> {standard_name}")
    
    def detect_column_type(self, df: pd.DataFrame, column: str) -> str:
        """
        Detect the likely type/purpose of a column based on its values.
        
        Args:
            df: DataFrame containing the column
            column: Column name to analyze
            
        Returns:
            Detected column type/purpose
        """
        if column not in df.columns:
            return "unknown"
        
        col_data = df[column]
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(col_data):
            # Check range to guess purpose
            if col_data.min() >= 0 and col_data.max() <= 100:
                return "percentage"
            elif col_data.min() < 0:
                return "balance_or_adjustment"
            else:
                return "quantity"
        
        # Check if date
        try:
            pd.to_datetime(col_data, errors='coerce')
            if col_data.notna().sum() > 0:
                return "date"
        except:
            pass
        
        # Check if identifier
        if col_data.dtype == 'object':
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio > 0.8:
                return "identifier"
            elif unique_ratio < 0.1:
                return "category"
            else:
                return "description"
        
        return "unknown"
    
    def suggest_standard_name(self, column: str, df: pd.DataFrame = None) -> Optional[str]:
        """
        Suggest a standard name for an unmapped column.
        
        Args:
            column: Column name to analyze
            df: Optional DataFrame for content analysis
            
        Returns:
            Suggested standard name or None
        """
        # First check if we already have a mapping
        if standard := self.get_standard_name(column):
            return standard
        
        # Try to guess based on column name patterns
        column_lower = column.lower()
        
        # Common patterns
        patterns = {
            'yarn_id': ['yarn', 'material', 'item'],
            'quantity': ['qty', 'amount', 'volume'],
            'date': ['date', 'time', 'when'],
            'status': ['status', 'state', 'condition'],
            'description': ['desc', 'name', 'title'],
            'cost': ['cost', 'price', 'value'],
            'balance': ['balance', 'remain', 'left']
        }
        
        for standard, keywords in patterns.items():
            if any(kw in column_lower for kw in keywords):
                return standard
        
        # If DataFrame provided, analyze content
        if df is not None and column in df.columns:
            col_type = self.detect_column_type(df, column)
            
            type_to_standard = {
                'percentage': 'percent',
                'balance_or_adjustment': 'balance',
                'quantity': 'quantity',
                'date': 'date',
                'identifier': 'id',
                'category': 'category',
                'description': 'description'
            }
            
            return type_to_standard.get(col_type)
        
        return None


class ValidationResult:
    """Result of column validation."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def __str__(self) -> str:
        result = []
        if self.errors:
            result.append(f"Errors: {', '.join(self.errors)}")
        if self.warnings:
            result.append(f"Warnings: {', '.join(self.warnings)}")
        return "; ".join(result) if result else "Valid"