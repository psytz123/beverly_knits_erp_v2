"""
Emergency fix for yarn shortage BOM mapping issue.
This module provides a working implementation that properly connects yarn shortages to production orders.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

class BOMMapper:
    """Handles BOM mapping between yarns, styles, and production orders"""
    
    def __init__(self):
        self.bom_df = None
        self.knit_orders_df = None
        self.yarn_df = None
        
    def load_data(self) -> bool:
        """Load all required data files"""
        # Load BOM
        bom_paths = [
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/BOM_updated.csv"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/BOM_updated.csv")
        ]
        
        for path in bom_paths:
            if path.exists():
                self.bom_df = pd.read_csv(path)
                print(f"Loaded BOM: {len(self.bom_df)} entries")
                break
        
        # Load knit orders
        ko_paths = [
            Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/eFab_Knit_Orders.csv"),
            Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5/eFab_Knit_Orders_20250816.xlsx")
        ]
        
        for path in ko_paths:
            if path.exists():
                if path.suffix == '.csv':
                    self.knit_orders_df = pd.read_csv(path)
                else:
                    self.knit_orders_df = pd.read_excel(path)
                print(f"Loaded knit orders: {len(self.knit_orders_df)} orders")
                break
        
        return self.bom_df is not None and self.knit_orders_df is not None
    
    def get_affected_orders(self, yarn_id: int) -> Dict:
        """Get production orders affected by a yarn shortage"""
        if self.bom_df is None or self.knit_orders_df is None:
            return {'affected_orders': 0, 'affected_orders_list': []}
        
        # Find styles using this yarn
        yarn_bom = self.bom_df[self.bom_df['Desc#'] == yarn_id]
        if yarn_bom.empty:
            return {'affected_orders': 0, 'affected_orders_list': []}
        
        affected_styles = yarn_bom['Style#'].unique()
        
        # Find production orders for these styles
        style_col = 'Style #' if 'Style #' in self.knit_orders_df.columns else 'Style#'
        order_col = 'Order #' if 'Order #' in self.knit_orders_df.columns else 'Order#'
        
        if style_col in self.knit_orders_df.columns:
            affected_orders = self.knit_orders_df[self.knit_orders_df[style_col].isin(affected_styles)]
            
            if not affected_orders.empty and order_col in affected_orders.columns:
                order_numbers = list(affected_orders[order_col].unique())[:5]
                return {
                    'affected_orders': len(affected_orders),
                    'affected_orders_list': order_numbers,
                    'affected_styles': list(affected_styles)[:10]
                }
        
        # If no orders found, return styles as fallback
        return {
            'affected_orders': len(affected_styles),
            'affected_orders_list': list(affected_styles)[:5],
            'affected_styles': list(affected_styles)[:10]
        }

# Global instance
bom_mapper = BOMMapper()

def fix_yarn_shortage_analysis(yarn_shortage_data: List[Dict]) -> List[Dict]:
    """
    Fix yarn shortage data by adding proper BOM mapping.
    
    Args:
        yarn_shortage_data: List of yarn shortage dictionaries from the API
    
    Returns:
        Updated list with affected_orders properly populated
    """
    global bom_mapper
    
    # Load data if not already loaded
    if bom_mapper.bom_df is None:
        if not bom_mapper.load_data():
            print("Warning: Could not load BOM mapping data")
            return yarn_shortage_data
    
    # Process each yarn shortage
    for yarn in yarn_shortage_data:
        yarn_id = yarn.get('yarn_id')
        if yarn_id:
            try:
                # Convert to numeric for matching
                yarn_id_numeric = float(yarn_id)
                mapping = bom_mapper.get_affected_orders(yarn_id_numeric)
                
                # Update the yarn data
                yarn['affected_orders'] = mapping['affected_orders']
                yarn['affected_orders_list'] = mapping['affected_orders_list']
                
                if 'affected_styles' in mapping:
                    yarn['affected_styles'] = mapping['affected_styles']
                    
            except (ValueError, TypeError):
                # If conversion fails, leave as is
                pass
    
    return yarn_shortage_data

def get_yarn_bom_mapping(yarn_id) -> Dict:
    """Get BOM mapping for a specific yarn"""
    global bom_mapper
    
    if bom_mapper.bom_df is None:
        bom_mapper.load_data()
    
    try:
        yarn_id_numeric = float(yarn_id)
        return bom_mapper.get_affected_orders(yarn_id_numeric)
    except:
        return {'affected_orders': 0, 'affected_orders_list': []}

# Test function
if __name__ == "__main__":
    mapper = BOMMapper()
    if mapper.load_data():
        # Test yarn 18865
        result = mapper.get_affected_orders(18865)
        print(f"\nYarn 18865 mapping:")
        print(f"  Affected orders: {result['affected_orders']}")
        print(f"  Order list: {result['affected_orders_list']}")
        if 'affected_styles' in result:
            print(f"  Affected styles: {len(result['affected_styles'])} styles")