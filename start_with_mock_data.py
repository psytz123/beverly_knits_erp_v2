#!/usr/bin/env python3
"""
Start Beverly Knits ERP with Mock Data
Forces the application to use mock data instead of production data
"""

import os
import sys

# Force mock data path
os.environ['DATA_BASE_DIR'] = '/mnt/c/finalee/beverly_knits_erp_v2/data/mock/5'
os.environ['MOCK_ERP_DATA'] = 'true'

# Modify data config before importing
config_file = '/mnt/c/finalee/beverly_knits_erp_v2/src/utils/data_config.py'

print("="*60)
print("🚀 STARTING BEVERLY KNITS ERP WITH MOCK DATA")
print("="*60)
print(f"Mock data path: {os.environ['DATA_BASE_DIR']}")
print("="*60 + "\n")

# Add src directories to path
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src')
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src/utils')
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src/core')

# Monkey patch the data config
import importlib.util
spec = importlib.util.spec_from_file_location("data_config", config_file)
data_config = importlib.util.module_from_spec(spec)
sys.modules['data_config'] = data_config
sys.modules['utils.data_config'] = data_config

# Override the data path
data_config.DATA_BASE_DIR = '/mnt/c/finalee/beverly_knits_erp_v2/data/mock/5'
data_config.DATA_FILES = {
    "sales": os.path.join(data_config.DATA_BASE_DIR, "Sales Activity Report.csv"),
    "yarn_inventory": os.path.join(data_config.DATA_BASE_DIR, "yarn_inventory.xlsx"),
    "knit_orders": os.path.join(data_config.DATA_BASE_DIR, "eFab_Knit_Orders.xlsx"),
    "bom": os.path.join(data_config.DATA_BASE_DIR, "BOM_updated.csv"),
    "inventory_f01": os.path.join(data_config.DATA_BASE_DIR, "ERP Data/eFab_Inventory_F01.xlsx"),
    "inventory_g02": os.path.join(data_config.DATA_BASE_DIR, "ERP Data/eFab_Inventory_G02.xlsx"),
    "inventory_i01": os.path.join(data_config.DATA_BASE_DIR, "ERP Data/eFab_Inventory_I01.xlsx"),
    "inventory_g00": os.path.join(data_config.DATA_BASE_DIR, "ERP Data/eFab_Inventory_G00.xlsx"),
}

# Load the spec
spec.loader.exec_module(data_config)

# Now import and run the main app
os.chdir('/mnt/c/finalee/beverly_knits_erp_v2')
from src.core.beverly_comprehensive_erp import app

print("\n✅ Mock data configuration loaded successfully!")
print("📊 Dashboard will be available at: http://localhost:5006/consolidated")
print("\n🌐 For ngrok sharing: ngrok http 5006")
print("="*60 + "\n")

# Run the app
app.run(host='0.0.0.0', port=5006, debug=False)