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
            
            # Create yarn netting data
            yarn_cols = ['Yarn ID', 'Description', 'Stock (LBS)']
            available_cols = [col for col in yarn_cols if col in analyzer.raw_materials_data.columns]
            
            if available_cols:
                yarn_data = analyzer.raw_materials_data[available_cols].head(20)
                for _, row in yarn_data.iterrows():
                    yarn_item = {
                        "yarn_id": str(row.get('Yarn ID', 'Unknown')),
                        "description": str(row.get('Description', 'N/A')),
                        "available_stock": float(row.get('Stock (LBS)', 0)) if not pd.isna(row.get('Stock (LBS)', 0)) else 0,
                        "allocated": 0,
                        "remaining": float(row.get('Stock (LBS)', 0)) if not pd.isna(row.get('Stock (LBS)', 0)) else 0
                    }
                    result['yarn_netting'].append(yarn_item)
        
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