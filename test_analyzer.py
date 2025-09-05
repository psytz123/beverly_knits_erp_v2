
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

# Set data path
DATA_PATH = Path(r'data\production\5\ERP Data')

print(f"Using data path: {DATA_PATH}")

# Import and initialize
try:
    from src.core.beverly_comprehensive_erp import ManufacturingSupplyChainAI
    
    # Create analyzer with correct data path
    analyzer = ManufacturingSupplyChainAI(DATA_PATH)
    
    # Check if data loaded
    if analyzer.raw_materials_data is not None:
        print(f"[OK] Yarn data loaded: {len(analyzer.raw_materials_data)} items")
    else:
        print("[FAIL] Yarn data not loaded")
    
    if analyzer.bom_data is not None:
        print(f"[OK] BOM data loaded: {len(analyzer.bom_data)} items")
    else:
        print("[FAIL] BOM data not loaded")
        
    # Test methods exist
    if hasattr(analyzer, 'get_advanced_inventory_optimization'):
        print("[OK] Advanced optimization method exists")
    else:
        print("[FAIL] Advanced optimization method missing")
        
except Exception as e:
    print(f"Error initializing analyzer: {e}")
    import traceback
    traceback.print_exc()
