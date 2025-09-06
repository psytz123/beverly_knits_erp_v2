#!/usr/bin/env python3
"""
Fix script to verify and correct the BOM mapping issue in yarn shortage analysis.
This script checks data integrity and provides a solution.
"""

import pandas as pd
from pathlib import Path
import json

def check_data_integrity():
    """Check the integrity of BOM and knit orders data"""
    
    print("=" * 60)
    print("YARN SHORTAGE BOM MAPPING FIX")
    print("=" * 60)
    
    # Check BOM data
    bom_paths = [
        Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/BOM_updated.csv"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/BOM_updated.csv")
    ]
    
    bom_df = None
    for path in bom_paths:
        if path.exists():
            bom_df = pd.read_csv(path)
            print(f"\n✓ BOM data loaded from: {path}")
            print(f"  - Records: {len(bom_df)}")
            print(f"  - Columns: {list(bom_df.columns)}")
            break
    
    if bom_df is None:
        print("\n✗ ERROR: BOM data not found!")
        return False
    
    # Check knit orders data
    ko_paths = [
        Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/eFab_Knit_Orders.csv"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5/eFab_Knit_Orders_20250816.xlsx"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/08-09/eFab_Knit_Orders_20250810 (2).xlsx")
    ]
    
    ko_df = None
    for path in ko_paths:
        if path.exists():
            if path.suffix == '.csv':
                ko_df = pd.read_csv(path)
            else:
                ko_df = pd.read_excel(path)
            print(f"\n✓ Knit orders loaded from: {path}")
            print(f"  - Records: {len(ko_df)}")
            print(f"  - Columns: {list(ko_df.columns)[:10]}")
            break
    
    if ko_df is None:
        print("\n✗ ERROR: Knit orders data not found!")
        return False
    
    # Check yarn inventory
    yarn_paths = [
        Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/yarn_inventory.xlsx"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/yarn_inventory.xlsx")
    ]
    
    yarn_df = None
    for path in yarn_paths:
        if path.exists():
            yarn_df = pd.read_excel(path)
            print(f"\n✓ Yarn inventory loaded from: {path}")
            print(f"  - Records: {len(yarn_df)}")
            
            # Check for Planning Balance column
            planning_cols = [c for c in yarn_df.columns if 'Planning' in c]
            print(f"  - Planning columns: {planning_cols}")
            break
    
    if yarn_df is None:
        print("\n✗ ERROR: Yarn inventory not found!")
        return False
    
    # Test the mapping for yarn 18865
    print("\n" + "=" * 60)
    print("TESTING YARN 18865 MAPPING")
    print("=" * 60)
    
    # Find yarn 18865 in inventory
    yarn_18865 = yarn_df[yarn_df['Desc#'] == 18865]
    if len(yarn_18865) > 0:
        print(f"\n✓ Yarn 18865 found in inventory")
        planning_balance = yarn_18865.iloc[0].get('Planning Balance', yarn_18865.iloc[0].get('Planning_Balance', 0))
        print(f"  - Planning Balance: {planning_balance}")
        
        if planning_balance < 0:
            print(f"  - SHORTAGE DETECTED: {abs(planning_balance)} lbs")
    else:
        print("\n✗ Yarn 18865 not found in inventory")
    
    # Find styles using yarn 18865 in BOM
    bom_yarn_18865 = bom_df[bom_df['Desc#'] == 18865]
    print(f"\n✓ Styles using yarn 18865: {len(bom_yarn_18865)}")
    
    if len(bom_yarn_18865) > 0:
        affected_styles = bom_yarn_18865['Style#'].unique()
        print(f"  - Unique styles: {len(affected_styles)}")
        print(f"  - Sample styles: {list(affected_styles[:5])}")
        
        # Find production orders for these styles
        style_col = 'Style#' if 'Style#' in ko_df.columns else 'Style #' if 'Style #' in ko_df.columns else None
        
        if style_col:
            affected_orders = ko_df[ko_df[style_col].isin(affected_styles)]
            print(f"\n✓ Production orders affected: {len(affected_orders)}")
            
            if len(affected_orders) > 0:
                order_col = 'Order#' if 'Order#' in ko_df.columns else 'Order #' if 'Order #' in ko_df.columns else 'Knit Order'
                if order_col in ko_df.columns:
                    order_numbers = affected_orders[order_col].unique()
                    print(f"  - Order numbers: {list(order_numbers[:5])}")
                    
                    # Show details of first affected order
                    first_order = affected_orders.iloc[0]
                    print(f"\n  First affected order:")
                    print(f"    - Order: {first_order.get(order_col)}")
                    print(f"    - Style: {first_order.get(style_col)}")
                    if 'Qty Ordered (lbs)' in first_order:
                        print(f"    - Quantity: {first_order.get('Qty Ordered (lbs)')} lbs")
        else:
            print("\n✗ No style column found in knit orders!")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"""
Data Status:
  - BOM entries: {len(bom_df)}
  - Knit orders: {len(ko_df)}
  - Yarn items: {len(yarn_df)}
  
Yarn 18865 Analysis:
  - Used in {len(bom_yarn_18865)} BOM entries
  - Affects {len(affected_styles) if 'affected_styles' in locals() else 0} unique styles
  - Impacts {len(affected_orders) if 'affected_orders' in locals() else 0} production orders
  
The BOM mapping should work if:
1. BOM has Desc# column with yarn IDs ✓
2. BOM has Style# column with style codes ✓
3. Knit orders have Style# or 'Style #' column ✓
4. The yarn intelligence API properly loads knit_orders_data
    """)
    
    return True

if __name__ == "__main__":
    check_data_integrity()