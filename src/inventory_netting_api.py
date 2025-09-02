"""
Inventory Netting API - Provides inventory allocation and netting analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_inventory_netting_endpoint(analyzer):
    """
    Create inventory netting analysis endpoint
    Allocates available inventory against demand and identifies gaps
    """
    try:
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "netting_summary": {
                "total_demand": 0,
                "total_inventory": 0,
                "total_allocated": 0,
                "total_shortage": 0,
                "fulfillment_rate": 0
            },
            "style_netting": [],
            "yarn_netting": [],
            "recommendations": []
        }
        
        # Get available data
        if hasattr(analyzer, 'sales_data') and analyzer.sales_data is not None and not analyzer.sales_data.empty:
            # Calculate total demand from sales data
            if 'Qty' in analyzer.sales_data.columns:
                total_demand = analyzer.sales_data['Qty'].sum()
                result['netting_summary']['total_demand'] = float(total_demand) if not pd.isna(total_demand) else 0
        
        if hasattr(analyzer, 'raw_materials_data') and analyzer.raw_materials_data is not None and not analyzer.raw_materials_data.empty:
            # Calculate total inventory
            if 'Stock (LBS)' in analyzer.raw_materials_data.columns:
                total_inventory = analyzer.raw_materials_data['Stock (LBS)'].sum()
                result['netting_summary']['total_inventory'] = float(total_inventory) if not pd.isna(total_inventory) else 0
            
            # Create yarn netting data with proper allocation
            # Check for correct column names
            stock_col = None
            id_col = None
            desc_col = None
            
            # Find the right columns
            if 'Planning Balance' in analyzer.raw_materials_data.columns:
                stock_col = 'Planning Balance'
            elif 'planning_balance' in analyzer.raw_materials_data.columns:
                stock_col = 'planning_balance'
            elif 'Stock (LBS)' in analyzer.raw_materials_data.columns:
                stock_col = 'Stock (LBS)'
            
            if 'Desc#' in analyzer.raw_materials_data.columns:
                id_col = 'Desc#'
            elif 'Yarn ID' in analyzer.raw_materials_data.columns:
                id_col = 'Yarn ID'
            elif 'yarn_id' in analyzer.raw_materials_data.columns:
                id_col = 'yarn_id'
            
            if 'Description' in analyzer.raw_materials_data.columns:
                desc_col = 'Description'
            elif 'description' in analyzer.raw_materials_data.columns:
                desc_col = 'description'
            
            if stock_col and id_col:
                # Get yarns sorted by stock level (prioritize low stock items)
                yarn_data = analyzer.raw_materials_data.sort_values(by=stock_col).head(50)
                
                # Calculate demand per yarn from BOM if available
                yarn_demand = {}
                if hasattr(analyzer, 'bom_data') and analyzer.bom_data is not None:
                    if 'Component' in analyzer.bom_data.columns and 'Quantity' in analyzer.bom_data.columns:
                        yarn_demand = analyzer.bom_data.groupby('Component')['Quantity'].sum().to_dict()
                
                for _, row in yarn_data.iterrows():
                    yarn_id = str(row.get(id_col, 'Unknown'))
                    available = float(row.get(stock_col, 0)) if not pd.isna(row.get(stock_col, 0)) else 0
                    demand = yarn_demand.get(yarn_id, 0)
                    
                    # Calculate allocation (cannot exceed available stock)
                    allocated = min(available, demand) if available > 0 else 0
                    remaining = available - allocated
                    
                    yarn_item = {
                        "yarn_id": yarn_id,
                        "description": str(row.get(desc_col, 'N/A')) if desc_col else 'N/A',
                        "available_stock": available,
                        "demand": demand,
                        "allocated": allocated,
                        "remaining": remaining,
                        "status": "SHORTAGE" if demand > available else "ADEQUATE"
                    }
                    result['yarn_netting'].append(yarn_item)
                
                # Update summary with actual allocated amount
                total_allocated = sum(item['allocated'] for item in result['yarn_netting'])
                result['netting_summary']['total_allocated'] = total_allocated
        
        # Calculate fulfillment rate
        if result['netting_summary']['total_demand'] > 0:
            result['netting_summary']['fulfillment_rate'] = min(
                100,
                (result['netting_summary']['total_inventory'] / result['netting_summary']['total_demand']) * 100
            )
        
        # Calculate allocated (simplified - just take minimum of demand and inventory)
        result['netting_summary']['total_allocated'] = min(
            result['netting_summary']['total_demand'],
            result['netting_summary']['total_inventory']
        )
        
        # Calculate shortage
        result['netting_summary']['total_shortage'] = max(
            0,
            result['netting_summary']['total_demand'] - result['netting_summary']['total_inventory']
        )
        
        # Add recommendations based on the analysis
        if result['netting_summary']['total_shortage'] > 0:
            result['recommendations'].append({
                "type": "shortage",
                "priority": "high",
                "message": f"Material shortage of {result['netting_summary']['total_shortage']:.0f} LBS detected. Consider expediting procurement."
            })
        
        if result['netting_summary']['fulfillment_rate'] < 80:
            result['recommendations'].append({
                "type": "fulfillment",
                "priority": "medium",
                "message": f"Fulfillment rate is {result['netting_summary']['fulfillment_rate']:.1f}%. Review production planning to improve availability."
            })
        
        if result['netting_summary']['total_inventory'] > result['netting_summary']['total_demand'] * 2:
            result['recommendations'].append({
                "type": "excess",
                "priority": "low",
                "message": "Excess inventory detected. Consider optimizing stock levels to reduce carrying costs."
            })
        
        return result
        
    except Exception as e:
        # Return a valid response even on error
        return {
            "status": "error",
            "error": str(e),
            "netting_summary": {
                "total_demand": 0,
                "total_inventory": 0,
                "total_allocated": 0,
                "total_shortage": 0,
                "fulfillment_rate": 0
            },
            "style_netting": [],
            "yarn_netting": [],
            "recommendations": [{
                "type": "error",
                "priority": "info",
                "message": "Unable to calculate netting due to data issues. Please check data availability."
            }]
        }