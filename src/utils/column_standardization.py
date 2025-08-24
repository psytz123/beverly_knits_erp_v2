"""
Column Standardization Module for Beverly Knits ERP System
Ensures consistent column naming across all data sources
"""

class ColumnStandardizer:
    """Standardizes column names across different data sources"""
    
    # Master column mapping dictionary
    COLUMN_MAPPINGS = {
        # Style-related columns
        'style_id': ['Style#', 'Style #', 'fStyle#', 'gStyle', 'Style_ID', 'style', 'Style'],
        
        # Yarn/Material identifiers
        'yarn_id': ['Desc#', 'desc#', 'Desc #', 'yarn_id', 'Yarn_ID', 'Material_ID'],
        
        # Descriptions
        'description': ['Description', 'Desc', 'description', 'desc', 'Material_Name'],
        
        # Quantity columns (lbs)
        'qty_lbs': ['Qty (lbs)', 'Quantity (lbs)', 'qty_lbs', 'Quantity_lbs'],
        'qty_ordered_lbs': ['Qty Ordered (lbs)', 'Ordered (lbs)', 'qty_ordered_lbs'],
        'balance_lbs': ['Balance (lbs)', 'balance_lbs', 'Balance_lbs'],
        'shipped_lbs': ['Shipped (lbs)', 'shipped_lbs', 'Shipped_lbs'],
        
        # Quantity columns (yards)
        'qty_yds': ['Qty (yds)', 'Quantity (yds)', 'qty_yds', 'Quantity_yds'],
        
        # Inventory balances
        'beginning_balance': ['Beginning Balance', 'beginning_balance', 'Starting_Balance'],
        'planning_balance': ['Planning Balance', 'planning_balance', 'Available_Balance'],
        'theoretical_balance': ['Theoretical Balance', 'theoretical_balance', 'Calculated_Balance'],
        
        # Date columns
        'start_date': ['Start Date', 'start_date', 'Begin_Date'],
        'ship_date': ['Ship Date', 'ship_date', 'Shipping_Date'],
        'quoted_date': ['Quoted Date', 'quoted_date', 'Quote_Date'],
        'po_date': ['PO Date', 'po_date', 'Purchase_Date'],
        'reconcile_date': ['Reconcile Date', 'reconcile_date', 'Reconciliation_Date'],
        
        # Order numbers
        'order_number': ['Order #', 'Order#', 'order_number', 'Order_Number'],
        'po_number': ['PO#', 'PO #', 'po_number', 'PO_Number'],
        'so_number': ['SO #', 'SO#', 'so_number', 'Sales_Order'],
        
        # Supplier information
        'supplier': ['Supplier', 'supplier', 'Vendor', 'vendor', 'Supplier_Name'],
        'supplier_id': ['Supplier_ID', 'supplier_id', 'Vendor_ID'],
        
        # Customer information
        'customer': ['Customer', 'customer', 'Client', 'client', 'Customer_Name'],
        
        # Cost/Price
        'cost_per_pound': ['Cost/Pound', 'cost_per_pound', 'Unit_Cost', 'Price_Per_Pound'],
        'unit_price': ['Unit Price', 'unit_price', 'Price', 'Unit_Price'],
        'total_cost': ['Total Cost', 'total_cost', 'Total_Value'],
        
        # Status fields
        'status': ['Status', 'status', 'Order_Status'],
        
        # BOM specific
        'bom_percentage': ['BOM_Percentage', 'bom_percentage', 'Percentage', 'Usage_Percentage'],
        
        # Inventory movements
        'received': ['Received', 'received', 'Receipts', 'Qty_Received'],
        'consumed': ['Consumed', 'consumed', 'Usage', 'Qty_Used'],
        'adjustments': ['Adjustments', 'adjustments', 'Adj', 'Inventory_Adj'],
        'on_order': ['On Order', 'on_order', 'Ordered', 'Qty_On_Order'],
        'allocated': ['Allocated', 'allocated', 'Reserved', 'Qty_Allocated'],
        
        # Location
        'location': ['Rack', 'rack', 'Location', 'location', 'Warehouse_Location'],
        
        # Machine/Equipment
        'machine': ['Machine', 'machine', 'Equipment', 'equipment'],
        
        # Color
        'color': ['Color', 'color', 'Colour', 'colour'],
        
        # Roll numbers
        'roll_number': ['Roll #', 'Roll#', 'roll_number', 'Roll_Number'],
        'vendor_roll_number': ['Vendor Roll #', 'vendor_roll_number', 'Supplier_Roll'],
        
        # Demand fields
        'total_demand': ['Total Demand', 'total_demand', 'Demand'],
        'total_receipt': ['Total Receipt', 'total_receipt', 'Expected_Receipt'],
        
        # Week-specific demands (for time-phased planning)
        'demand_this_week': ['Demand This Week', 'This Week', 'demand_this_week'],
        'receipts_this_week': ['Receipts This Week', 'receipts_this_week'],
        'balance_this_week': ['Balance This Week', 'balance_this_week'],
        
        # Inventory stages
        'g00_lbs': ['G00 (lbs)', 'g00_lbs', 'Stage_G00'],
        'g02_lbs': ['G02 (lbs)', 'g02_lbs', 'Stage_G02'],
        'f01_lbs': ['F01 (lbs)', 'f01_lbs', 'Stage_F01'],
        'i01_lbs': ['I01 (lbs)', 'i01_lbs', 'Stage_I01'],
        
        # Quality
        'good_qty': ['Good Ea.', 'good_qty', 'Good_Quantity'],
        'bad_qty': ['Bad Ea.', 'bad_qty', 'Defect_Quantity'],
        'seconds_lbs': ['Seconds (lbs)', 'seconds_lbs', 'Second_Quality'],
        
        # Other common fields
        'uom': ['UOM', 'uom', 'Unit', 'unit', 'Unit_Of_Measure'],
        'lead_time': ['Lead_time', 'lead_time', 'Lead_Time', 'LeadTime'],
        'moq': ['MOQ', 'moq', 'Min_Order_Qty', 'Minimum_Order'],
        'type': ['Type', 'type', 'Material_Type', 'Product_Type'],
        'blend': ['Blend', 'blend', 'Material_Blend', 'Composition'],
        'ply': ['Ply', 'ply', 'Yarn_Ply'],
        'size': ['Size', 'size', 'Yarn_Size'],
        'filament': ['Filament', 'filament', 'Yarn_Filament'],
    }
    
    @classmethod
    def standardize_dataframe(cls, df, file_type=None):
        """
        Standardize column names in a dataframe
        
        Args:
            df: pandas DataFrame to standardize
            file_type: Optional string indicating the type of file 
                      (e.g., 'yarn_inventory', 'knit_orders', etc.)
        
        Returns:
            DataFrame with standardized column names
        """
        import pandas as pd
        
        # Create a copy to avoid modifying the original
        df_standard = df.copy()
        
        # Create reverse mapping for efficient lookups
        reverse_mapping = {}
        for standard_name, variations in cls.COLUMN_MAPPINGS.items():
            for variation in variations:
                reverse_mapping[variation] = standard_name
        
        # Rename columns based on mapping
        rename_dict = {}
        for col in df_standard.columns:
            if col in reverse_mapping:
                rename_dict[col] = reverse_mapping[col]
        
        if rename_dict:
            df_standard = df_standard.rename(columns=rename_dict)
            
        # Apply file-specific standardizations if needed
        if file_type:
            df_standard = cls._apply_file_specific_rules(df_standard, file_type)
        
        return df_standard
    
    @classmethod
    def _apply_file_specific_rules(cls, df, file_type):
        """Apply file-specific standardization rules"""
        
        if file_type == 'yarn_inventory':
            # Ensure key columns exist for yarn inventory
            required_cols = ['yarn_id', 'description', 'planning_balance']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in yarn_inventory")
                    
        elif file_type == 'knit_orders':
            # Ensure key columns exist for knit orders
            required_cols = ['style_id', 'order_number', 'qty_ordered_lbs', 'balance_lbs']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in knit_orders")
                    
        elif file_type == 'bom':
            # Ensure BOM has style and yarn mappings
            required_cols = ['style_id', 'yarn_id', 'bom_percentage']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in BOM")
        
        return df
    
    @classmethod
    def get_standard_name(cls, column_name):
        """Get the standardized name for a given column"""
        for standard_name, variations in cls.COLUMN_MAPPINGS.items():
            if column_name in variations:
                return standard_name
        return column_name  # Return original if no mapping found
    
    @classmethod
    def validate_required_columns(cls, df, required_columns):
        """
        Validate that required columns exist in the dataframe
        
        Args:
            df: pandas DataFrame to validate
            required_columns: list of required standard column names
        
        Returns:
            tuple (is_valid, missing_columns)
        """
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing


# Example usage function
def standardize_all_data_files(data_path):
    """
    Standardize all data files in the specified directory
    
    Args:
        data_path: Path to the data directory
    """
    import pandas as pd
    from pathlib import Path
    
    data_path = Path(data_path)
    standardizer = ColumnStandardizer()
    
    # Define file mappings
    file_mappings = {
        'yarn_inventory (2).xlsx': 'yarn_inventory',
        'eFab_Knit_Orders_20250810 (2).xlsx': 'knit_orders',
        'eFab_SO_List_202508101032.xlsx': 'sales_orders',
        'Style_BOM.csv': 'bom',
        'Yarn_Demand_2025-08-09_0442.xlsx': 'yarn_demand',
        'Yarn_Demand_By_Style.xlsx': 'yarn_demand_by_style',
    }
    
    standardized_data = {}
    
    for filename, file_type in file_mappings.items():
        file_path = data_path / filename
        if file_path.exists():
            # Load file
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Standardize columns
            df_standard = standardizer.standardize_dataframe(df, file_type)
            standardized_data[file_type] = df_standard
            
            print(f"Standardized {file_type}:")
            print(f"  Original columns: {len(df.columns)}")
            print(f"  Standardized columns: {list(df_standard.columns)[:5]}...")
            print()
    
    return standardized_data