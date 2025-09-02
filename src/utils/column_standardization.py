"""
Column Standardization Module for Beverly Knits ERP System
Ensures consistent column naming across all data sources
"""

class ColumnStandardizer:
    """Standardizes column names across different data sources based on actual file headers"""
    
    # Master column mapping dictionary - Maps to ACTUAL column names used in data files
    COLUMN_MAPPINGS = {
        # Yarn identifiers - standardize to 'Desc#' as used in most files
        'Desc#': ['Desc#', 'desc#', 'Desc #', 'Yarn', 'yarn', 'Yarn_ID', 'YarnID', 'yarn_id', 'Material_ID', 'Desc'],
        
        # Style-related columns - keep variations as they appear in files
        'Style#': ['Style#', 'Style #', 'Style', 'style', 'style_id', 'Style_ID'],
        'fStyle#': ['fStyle#', 'fStyle'],  # Don't include 'Style #' here - it maps to Style# in most cases
        'gStyle': ['gStyle', 'gStyle#'],
        
        # Descriptions
        'Description': ['Description', 'Desc', 'description', 'desc', 'Material_Name'],
        
        # Quantity columns (lbs) - as they appear in knit orders
        'Qty Ordered (lbs)': ['Qty Ordered (lbs)', 'qty_ordered_lbs', 'Ordered (lbs)'],
        'Balance (lbs)': ['Balance (lbs)', 'balance_lbs', 'Balance_lbs'],
        'Shipped (lbs)': ['Shipped (lbs)', 'shipped_lbs', 'Shipped_lbs'],
        'G00 (lbs)': ['G00 (lbs)', 'g00_lbs', 'Stage_G00'],
        'Seconds (lbs)': ['Seconds (lbs)', 'seconds_lbs', 'Second_Quality'],
        
        # Quantity columns (yards)
        'Qty (yds)': ['Qty (yds)', 'Quantity (yds)', 'qty_yds', 'Quantity_yds'],
        'Qty (lbs)': ['Qty (lbs)', 'Quantity (lbs)', 'qty_lbs', 'Quantity_lbs'],
        
        # Inventory balances - handle all variations including typos
        'Beginning Balance': ['Beginning Balance', 'beginning_balance', 'Starting_Balance'],
        'Planning Balance': ['Planning Balance', 'Planning_Balance', 'Planning_Ballance', 'planning_balance', 'Available_Balance'],
        'Theoretical Balance': ['Theoretical Balance', 'theoretical_balance', 'Calculated_Balance'],
        
        # Date columns
        'Start Date': ['Start Date', 'start_date', 'Begin_Date'],
        'Ship Date': ['Ship Date', 'ship_date', 'Shipping_Date'],
        'Quoted Date': ['Quoted Date', 'quoted_date', 'Quote_Date'],
        'PO Date': ['PO Date', 'po_date', 'Purchase_Date'],
        'Reconcile Date': ['Reconcile Date', 'reconcile_date', 'Reconciliation_Date'],
        'Knit Date': ['Knit Date', 'knit_date'],
        
        # Order numbers - as they appear in files
        'Order #': ['Order #', 'Order#', 'order_number', 'Order_Number'],
        'PO#': ['PO#', 'PO #', 'po_number', 'PO_Number'],
        'SO #': ['SO #', 'SO#', 'so_number', 'Sales_Order'],
        'KO #': ['KO #', 'KO#', 'Actions'],  # Knit Order number
        
        # Supplier information
        'Supplier': ['Supplier', 'supplier', 'Vendor', 'vendor', 'Supplier_Name', 'Purchased From'],
        'Supplier_ID': ['Supplier_ID', 'supplier_id', 'Vendor_ID'],
        
        # Customer information
        'Customer': ['Customer', 'customer', 'Client', 'client', 'Customer_Name', 'Sold To'],
        'Ship To': ['Ship To', 'ship_to'],
        
        # Cost/Price
        'Cost/Pound': ['Cost/Pound', 'Cost_Pound', 'cost_per_pound', 'Unit_Cost', 'Price_Per_Pound'],
        'Unit Price': ['Unit Price', 'unit_price', 'Price', 'Unit_Price'],
        'Total Cost': ['Total Cost', 'Total_Cost', 'Total_Cast', 'total_cost', 'Total_Value'],  # Note: Total_Cast typo
        'Line Price': ['Line Price', 'line_price'],
        
        # Status fields
        'Status': ['Status', 'status', 'Order_Status'],
        
        # BOM specific - handle both variations found in files
        'BOM_Percent': ['BOM_Percent', 'BOM_Percentage', 'Percentage', 'BOM%', 'bom_percentage', 'Usage_Percentage'],
        
        # Inventory movements - handle space vs underscore variations
        'Received': ['Received', 'received', 'Receipts', 'Qty_Received'],
        'Consumed': ['Consumed', 'consumed', 'Usage', 'Qty_Used'],
        'Adjustments': ['Adjustments', 'adjustments', 'Adj', 'Inventory_Adj'],
        'On Order': ['On Order', 'On_Order', 'on_order', 'Ordered', 'Qty_On_Order'],
        'Allocated': ['Allocated', 'allocated', 'Reserved', 'Qty_Allocated'],
        
        # Location
        'Rack': ['Rack', 'rack', 'Location', 'location', 'Warehouse_Location'],
        
        # Machine/Equipment
        'Machine': ['Machine', 'machine', 'Equipment', 'equipment'],
        
        # Color
        'Color': ['Color', 'color', 'Colour', 'colour'],
        
        # Roll numbers
        'Roll #': ['Roll #', 'Roll#', 'roll_number', 'Roll_Number'],
        'Vendor Roll #': ['Vendor Roll #', 'vendor_roll_number', 'Supplier_Roll'],
        
        # Demand fields (from Yarn_Demand files)
        'Total Demand': ['Total Demand', 'total_demand', 'Demand'],
        'Total Receipt': ['Total Receipt', 'total_receipt', 'Expected_Receipt'],
        'Monday Inventory': ['Monday Inventory', 'monday_inventory'],
        
        # Week-specific demands (for time-phased planning)
        'Demand This Week': ['Demand This Week', 'This Week', 'demand_this_week'],
        'Receipts This Week': ['Receipts This Week', 'receipts_this_week'],
        'Balance This Week': ['Balance This Week', 'balance_this_week'],
        
        # Quality fields
        'Good Ea.': ['Good Ea.', 'good_qty', 'Good_Quantity'],
        'Bad Ea.': ['Bad Ea.', 'bad_qty', 'Defect_Quantity'],
        
        # Other common fields
        'UOM': ['UOM', 'uom', 'Unit', 'unit', 'Unit_Of_Measure'],
        'Lead_time': ['Lead_time', 'lead_time', 'Lead_Time', 'LeadTime'],
        'MOQ': ['MOQ', 'moq', 'Min_Order_Qty', 'Minimum_Order'],
        'Type': ['Type', 'type', 'Material_Type', 'Product_Type'],
        'Blend': ['Blend', 'blend', 'Material_Blend', 'Composition'],
        'Ply': ['Ply', 'ply', 'Yarn_Ply'],
        'Size': ['Size', 'size', 'Yarn_Size'],
        'Filament': ['Filament', 'filament', 'Yarn_Filament'],
        
        # Sales Activity Report specific
        'Document': ['Document', 'document', 'Doc_Number'],
        'Invoice Date': ['Invoice Date', 'invoice_date'],
        'Yds_ordered': ['Yds_ordered', 'yds_ordered', 'Yards_Ordered'],
        'Sales Rep': ['Sales Rep', 'sales_rep'],
        
        # QuadS fabric specifications
        'F ID': ['F ID', 'f_id', 'Fabric_ID'],
        'Finish Code': ['Finish Code', 'finish_code'],
        'Overall Width': ['Overall Width', 'overall_width'],
        'Cuttable Width': ['Cuttable Width', 'cuttable_width'],
        'Oz/Lin Yd': ['Oz/Lin Yd', 'oz_lin_yd'],
        'Yds/Lbs': ['Yds/Lbs', 'yds_lbs', 'Yards_Per_Pound'],
        'GSM': ['GSM', 'gsm', 'Grams_Per_SqM'],
        
        # Additional fields from actual files
        'QS Cust': ['QS Cust', 'qs_cust'],
        'Misc': ['Misc', 'misc', 'Miscellaneous'],
        'BKI #s': ['BKI #s', 'bki_numbers'],
        'Modified': ['Modified', 'modified', 'Last_Modified'],
        'CSR': ['CSR', 'csr', 'Customer_Service_Rep'],
        'cFVersion': ['cFVersion', 'cf_version'],
        'fBase': ['fBase', 'f_base'],
        'On Hold': ['On Hold', 'on_hold'],
        'Picked/Shipped': ['Picked/Shipped', 'picked_shipped'],
        'Available': ['Available', 'available'],
        'SOP': ['SOP', 'sop'],
        'DO': ['DO', 'do', 'Dye_Order'],
        'Processor': ['Processor', 'processor'],
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
        """Apply file-specific standardization rules based on actual file types"""
        
        if file_type == 'yarn_inventory':
            # yarn_inventory.xlsx already has correct columns
            required_cols = ['Desc#', 'Description', 'Planning Balance']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in yarn_inventory")
                    
        elif file_type == 'yarn_id':
            # Yarn_ID.csv and Yarn_ID_Master.csv have Planning_Ballance typo
            if 'Planning_Ballance' in df.columns and 'Planning Balance' not in df.columns:
                df = df.rename(columns={'Planning_Ballance': 'Planning Balance'})
            if 'On_Order' in df.columns and 'On Order' not in df.columns:
                df = df.rename(columns={'On_Order': 'On Order'})
                    
        elif file_type == 'knit_orders' or file_type == 'eFab_Knit_Orders':
            # eFab_Knit_Orders has 'Style #' with space - keep it as Style# not fStyle#
            if 'Style #' in df.columns and 'Style#' not in df.columns:
                df = df.rename(columns={'Style #': 'Style#'})
            # Don't map to fStyle# for knit orders
            if 'fStyle#' in df.columns and 'Style#' not in df.columns:
                df = df.rename(columns={'fStyle#': 'Style#'})
            required_cols = ['Style#', 'Order #', 'Qty Ordered (lbs)', 'Balance (lbs)']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in knit_orders")
                    
        elif file_type == 'bom':
            # BOM files - standardize BOM_Percentage to BOM_Percent
            if 'BOM_Percentage' in df.columns and 'BOM_Percent' not in df.columns:
                df = df.rename(columns={'BOM_Percentage': 'BOM_Percent'})
            required_cols = ['Style#', 'Desc#', 'BOM_Percent']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found in BOM")
                    
        elif file_type == 'inventory_f01' or file_type == 'inventory_g00' or file_type == 'inventory_i01':
            # Inventory files have 'Style #' with space, should map to fStyle#
            if 'Style #' in df.columns and 'fStyle#' not in df.columns:
                df = df.rename(columns={'Style #': 'fStyle#'})
                
        elif file_type == 'inventory_g02':
            # G02 has unique 'fStyle' and 'gStyle' without #
            if 'fStyle' in df.columns and 'fStyle#' not in df.columns:
                df = df.rename(columns={'fStyle': 'fStyle#'})
            if 'gStyle' in df.columns and 'gStyle#' not in df.columns:
                df = df.rename(columns={'gStyle': 'gStyle#'})
                
        elif file_type == 'sales_activity':
            # Sales Activity Report already has fStyle#, which is correct
            pass
            
        elif file_type == 'so_list':
            # eFab_SO_List is missing style columns - log warning
            if 'fStyle#' not in df.columns and 'Style#' not in df.columns:
                print("WARNING: eFab_SO_List is missing style columns (fStyle# and Style#)")
                
        elif file_type == 'yarn_demand' or file_type == 'yarn_demand_by_style':
            # Yarn Demand files use 'Yarn' instead of 'Desc#'
            if 'Yarn' in df.columns and 'Desc#' not in df.columns:
                df = df.rename(columns={'Yarn': 'Desc#'})
        
        return df
    
    @classmethod
    def get_standard_name(cls, column_name):
        """Get the standardized name for a given column"""
        for standard_name, variations in cls.COLUMN_MAPPINGS.items():
            if column_name in variations:
                return standard_name
        return column_name  # Return original if no mapping found
    
    @classmethod
    def find_column(cls, df, variations):
        """
        Find first matching column from list of variations in a dataframe
        
        Args:
            df: pandas DataFrame to search
            variations: list of column name variations to try
        
        Returns:
            str: First matching column name found, or None if no match
        """
        if hasattr(df, 'columns'):
            for col in variations:
                if col in df.columns:
                    return col
        return None
    
    @classmethod
    def find_column_value(cls, row, variations, default=None):
        """
        Get value from row using first matching column variation
        
        Args:
            row: pandas Series or dict-like row
            variations: list of column name variations to try
            default: default value if no column found
        
        Returns:
            Value from first matching column, or default
        """
        for col in variations:
            if col in row:
                return row[col]
        return default
    
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
    
    def standardize_columns(self, df, file_type=None):
        """
        Alias for standardize_dataframe for backward compatibility
        
        Args:
            df: pandas DataFrame to standardize
            file_type: Optional string indicating the type of file
        
        Returns:
            DataFrame with standardized column names
        """
        return self.standardize_dataframe(df, file_type)


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