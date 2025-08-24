# Data file configuration for Beverly Knits ERP
# Points to the latest data files in prompts/5 directory

import os

# Base data directory
DATA_BASE_DIR = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5"

# Define file paths for each data type
DATA_FILES = {
    "sales": os.path.join(DATA_BASE_DIR, "eFab_SO_List_202508101032.xlsx"),
    "yarn_inventory": os.path.join(DATA_BASE_DIR, "yarn_inventory (2).xlsx"),
    "knit_orders": os.path.join(DATA_BASE_DIR, "eFab_Knit_Orders_20250810 (2).xlsx"),
    "bom": os.path.join(DATA_BASE_DIR, "Style_BOM.csv"),
    "yarn_demand": os.path.join(DATA_BASE_DIR, "Yarn_Demand_2025-08-09_0442.xlsx"),
    "yarn_demand_by_style": os.path.join(DATA_BASE_DIR, "Yarn_Demand_By_Style.xlsx"),
    "expected_yarn": os.path.join(DATA_BASE_DIR, "Expected_Yarn_Report.xlsx"),
    "supplier": os.path.join(DATA_BASE_DIR, "Supplier_ID.csv"),
    "yarn_id": os.path.join(DATA_BASE_DIR, "Yarn_ID_1.csv"),
    "inventory_f01": os.path.join(DATA_BASE_DIR, "eFab_Inventory_F01_20250810.xlsx"),
    "inventory_g02": os.path.join(DATA_BASE_DIR, "eFab_Inventory_G02_20250810.xlsx"),
    "inventory_i01": os.path.join(DATA_BASE_DIR, "eFab_Inventory_I01_20250810 (1).xlsx")
}

# Column mappings for standardization
COLUMN_MAPPINGS = {
    "sales": {
        "style_col": "fStyle#",
        "qty_ordered_col": "Ordered",
        "qty_shipped_col": "Picked/Shipped",
        "date_col": "Quoted Date",
        "price_col": "Unit Price",
        "customer_col": "Sold To"
    },
    "yarn_inventory": {
        "yarn_id_col": "Desc#",
        "balance_col": "Planning Balance",
        "on_order_col": "On Order",
        "cost_col": "Cost/Pound",
        "supplier_col": "Supplier"
    },
    "bom": {
        "style_col": "Style#",
        "yarn_id_col": "desc#",
        "percentage_col": "BOM_Percentage",
        "unit_col": "unit"
    }
}

print(f"Data configuration loaded from: {DATA_BASE_DIR}")
