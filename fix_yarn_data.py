#!/usr/bin/env python3
"""
Yarn Data Cleaning and Standardization Script
Fixes issues with yarn inventory data formatting and BOM mapping consistency
"""

import pandas as pd
import os
import re
import shutil
from datetime import datetime

def clean_currency_column(series):
    """Clean currency columns by removing $ signs and commas"""
    if series.dtype == 'object':
        # Remove $ signs, commas, and extra spaces
        return series.astype(str).str.replace(r'[\$,\s"]', '', regex=True)
    return series

def clean_yarn_inventory_file(filepath):
    """Clean and standardize yarn inventory file"""
    print(f"Processing: {filepath}")
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Read with proper encoding handling
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Successfully read with {encoding} encoding")
            break
        except Exception as e:
            print(f"Failed with {encoding}: {e}")
            continue
    
    if df is None:
        raise Exception("Could not read file with any encoding")
    
    # Clean column names (remove BOM character)
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    
    # Standardize numeric columns
    numeric_columns = ['Beginning Balance', 'Received', 'Consumed', 'Adjustments', 
                      'Theoretical Balance', 'Misc', 'On Order', 'Allocated', 
                      'Planning Balance', 'Cost/Pound', 'Total Cost']
    
    for col in numeric_columns:
        if col in df.columns:
            # Clean currency formatting
            df[col] = clean_currency_column(df[col])
            # Convert to numeric, handling errors
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Clean Description column (remove extra spaces)
    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype(str).str.strip()
        # Remove trailing commas from description
        df['Description'] = df['Description'].str.rstrip(',')
    
    # Standardize color names
    if 'Color' in df.columns:
        df['Color'] = df['Color'].astype(str).str.strip()
    
    # Ensure Desc# is string and clean
    if 'Desc#' in df.columns:
        df['Desc#'] = df['Desc#'].astype(str).str.strip()
    
    # Save cleaned file
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Cleaned file saved: {filepath}")
    
    return df

def validate_bom_mappings(bom_path, yarn_path):
    """Validate BOM mappings against yarn inventory"""
    print(f"\nValidating BOM mappings...")
    
    # Read BOM data
    bom_df = pd.read_csv(bom_path)
    yarn_df = pd.read_csv(yarn_path)
    
    # Get unique yarn IDs from both files
    bom_yarns = set(bom_df['Desc#'].astype(str))
    inventory_yarns = set(yarn_df['Desc#'].astype(str))
    
    # Find mismatches
    missing_in_inventory = bom_yarns - inventory_yarns
    missing_in_bom = inventory_yarns - bom_yarns
    
    print(f"BOM yarn entries: {len(bom_yarns)}")
    print(f"Inventory yarn entries: {len(inventory_yarns)}")
    print(f"Yarns in BOM but not in inventory: {len(missing_in_inventory)}")
    print(f"Yarns in inventory but not in BOM: {len(missing_in_bom)}")
    
    if missing_in_inventory:
        print(f"\nFirst 10 missing from inventory: {list(missing_in_inventory)[:10]}")
    
    return {
        'bom_yarns': bom_yarns,
        'inventory_yarns': inventory_yarns,
        'missing_in_inventory': missing_in_inventory,
        'missing_in_bom': missing_in_bom
    }

def main():
    """Main execution function"""
    base_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data"
    
    # Files to clean
    yarn_files = [
        os.path.join(base_path, "yarn_inventory.csv"),
        os.path.join(base_path, "8-28-2025", "yarn_inventory.csv")
    ]
    
    bom_file = os.path.join(base_path, "BOM_updated.csv")
    
    print("=== YARN DATA CLEANING SCRIPT ===\n")
    
    # Clean yarn inventory files
    for yarn_file in yarn_files:
        if os.path.exists(yarn_file):
            try:
                clean_yarn_inventory_file(yarn_file)
                print(f"✅ Successfully cleaned {yarn_file}")
            except Exception as e:
                print(f"❌ Error cleaning {yarn_file}: {e}")
        else:
            print(f"⚠️  File not found: {yarn_file}")
    
    # Validate BOM mappings
    if os.path.exists(bom_file) and os.path.exists(yarn_files[0]):
        try:
            validation_results = validate_bom_mappings(bom_file, yarn_files[0])
            print("✅ BOM validation completed")
        except Exception as e:
            print(f"❌ Error validating BOM: {e}")
    
    print("\n=== YARN DATA CLEANING COMPLETED ===")

if __name__ == "__main__":
    main()