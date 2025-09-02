#!/usr/bin/env python3
"""
Load 9-2-2025 ERP data into the system.
"""

import os
import sys
import pandas as pd
import shutil
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.cache_manager import CacheManager

# Configuration
DATA_DIR = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data'
NEW_DATA_DIR = os.path.join(DATA_DIR, '9-2-2025')
BACKUP_DIR = os.path.join(DATA_DIR, f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

# Files to load
FILES_TO_LOAD = [
    'yarn_inventory.xlsx',
    'eFab_Knit_Orders.xlsx', 
    'eFab_SO_List.xlsx',
    'Yarn_Demand.xlsx',
    'Yarn_Demand_By_Style.xlsx',
    'Yarn_Demand_By_Style_KO.xlsx',
    'Expected_Yarn_Report.xlsx',
    'YarnPO.xlsx',
    'eFab_Inventory_F01.xlsx',
    'eFab_Inventory_G00.xlsx',
    'eFab_Inventory_G02.xlsx',
    'eFab_Inventory_I01.xlsx'
]

def backup_existing_data():
    """Backup existing data files."""
    print(f"\n📦 Creating backup in {BACKUP_DIR}")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    for file in FILES_TO_LOAD:
        src_file = os.path.join(DATA_DIR, file)
        csv_file = src_file.replace('.xlsx', '.csv')
        
        # Backup Excel if exists
        if os.path.exists(src_file):
            dst_file = os.path.join(BACKUP_DIR, file)
            shutil.copy2(src_file, dst_file)
            print(f"  ✓ Backed up {file}")
        
        # Backup CSV if exists
        if os.path.exists(csv_file):
            dst_csv = os.path.join(BACKUP_DIR, os.path.basename(csv_file))
            shutil.copy2(csv_file, dst_csv)
            print(f"  ✓ Backed up {os.path.basename(csv_file)}")

def convert_and_load_data():
    """Convert Excel files to CSV and copy to main directory."""
    print(f"\n📊 Loading data from {NEW_DATA_DIR}")
    
    loaded_files = []
    errors = []
    
    for file in FILES_TO_LOAD:
        src_file = os.path.join(NEW_DATA_DIR, file)
        dst_excel = os.path.join(DATA_DIR, file)
        dst_csv = os.path.join(DATA_DIR, file.replace('.xlsx', '.csv'))
        
        try:
            if os.path.exists(src_file):
                # Read Excel file
                print(f"\n  Processing {file}...")
                df = pd.read_excel(src_file)
                
                # Show file info
                print(f"    Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                print(f"    Columns: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"    Columns: {', '.join(df.columns)}")
                
                # Save as CSV
                df.to_csv(dst_csv, index=False)
                print(f"    ✓ Saved as CSV: {os.path.basename(dst_csv)}")
                
                # Copy Excel file to main directory
                shutil.copy2(src_file, dst_excel)
                print(f"    ✓ Copied Excel to main directory")
                
                loaded_files.append(file)
            else:
                print(f"  ⚠️ File not found: {file}")
                
        except Exception as e:
            errors.append(f"{file}: {str(e)}")
            print(f"  ❌ Error processing {file}: {str(e)}")
    
    return loaded_files, errors

def clear_cache():
    """Clear the cache to force reload of new data."""
    print("\n🧹 Clearing cache...")
    try:
        cache_manager = CacheManager()
        cache_manager.clear_all()
        print("  ✓ Cache cleared successfully")
    except Exception as e:
        print(f"  ⚠️ Could not clear cache: {str(e)}")
    
    # Also clear file-based cache
    cache_dir = '/tmp/bki_cache'
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("  ✓ File cache cleared")
        except Exception as e:
            print(f"  ⚠️ Could not clear file cache: {str(e)}")

def verify_data_integrity():
    """Verify that critical data files are loaded correctly."""
    print("\n🔍 Verifying data integrity...")
    
    critical_files = [
        ('yarn_inventory.csv', ['Planning Balance', 'Planning_Balance']),
        ('eFab_Knit_Orders.csv', ['Machine', 'Style#', 'fStyle#']),
        ('Yarn_Demand_By_Style_KO.csv', ['Style#', 'fStyle#', 'Desc#'])
    ]
    
    all_good = True
    for file, expected_cols in critical_files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                found_cols = [col for col in expected_cols if col in df.columns]
                if found_cols:
                    print(f"  ✓ {file}: Found columns {found_cols}")
                else:
                    print(f"  ⚠️ {file}: Missing expected columns. Has: {list(df.columns)[:5]}...")
                    all_good = False
            except Exception as e:
                print(f"  ❌ {file}: Error reading - {str(e)}")
                all_good = False
        else:
            print(f"  ⚠️ {file}: File not found")
            all_good = False
    
    return all_good

def main():
    """Main execution."""
    print("="*60)
    print("📅 LOADING 9-2-2025 ERP DATA")
    print("="*60)
    
    # Check if new data directory exists
    if not os.path.exists(NEW_DATA_DIR):
        print(f"❌ Error: Directory not found: {NEW_DATA_DIR}")
        return 1
    
    # Step 1: Backup existing data
    backup_existing_data()
    
    # Step 2: Load new data
    loaded_files, errors = convert_and_load_data()
    
    # Step 3: Clear cache
    clear_cache()
    
    # Step 4: Verify data integrity
    integrity_ok = verify_data_integrity()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"✅ Successfully loaded: {len(loaded_files)} files")
    if loaded_files:
        for f in loaded_files:
            print(f"  • {f}")
    
    if errors:
        print(f"\n❌ Errors encountered: {len(errors)}")
        for e in errors:
            print(f"  • {e}")
    
    if integrity_ok:
        print("\n✅ Data integrity check passed")
    else:
        print("\n⚠️ Data integrity check found issues - review above")
    
    print("\n💡 Next steps:")
    print("  1. Restart the ERP server to load new data:")
    print("     pkill -f 'python3.*beverly' && python3 src/core/beverly_comprehensive_erp.py")
    print("  2. Visit http://localhost:5006/consolidated to verify")
    print("  3. Check API: curl http://localhost:5006/api/debug-data | python3 -m json.tool")
    
    return 0 if (loaded_files and integrity_ok) else 1

if __name__ == '__main__':
    sys.exit(main())