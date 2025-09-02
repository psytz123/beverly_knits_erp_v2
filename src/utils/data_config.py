# Data file configuration for Beverly Knits ERP
# Integrated with PostgreSQL database and centralized data source

import os
import json
from pathlib import Path

# Load unified configuration
config_path = Path(__file__).parent.parent / 'config' / 'unified_config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        unified_config = json.load(f)
else:
    unified_config = {"data_source": {"files": {"primary_path": "/mnt/c/Users/psytz/sc data/ERP Data"}}}

# Base data directory - now points to centralized location
DATA_BASE_DIR = unified_config["data_source"]["files"]["primary_path"]

# Fallback to latest data if primary path doesn't have files
if not os.path.exists(DATA_BASE_DIR) or not os.listdir(DATA_BASE_DIR):
    # Try to find most recent data folder
    for date_folder in ['8-28-2025', '8-26-2025', '8-24-2025', '8-22-2025']:
        test_path = os.path.join(DATA_BASE_DIR, date_folder)
        if os.path.exists(test_path):
            DATA_BASE_DIR = test_path
            break

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
