"""
Fixed Production Suggestions API - Uses real data to generate suggestions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

def get_production_suggestions_fixed(data_loader) -> Dict[str, Any]:
    """
    Generate production suggestions based on real orders and inventory
    """
    try:
        if not data_loader or not hasattr(data_loader, 'data'):
            return {"status": "error", "message": "Data loader not available"}
        
        # Get required data
        knit_orders = data_loader.data.get('knit_orders')
        yarn_inventory = data_loader.data.get('yarn_inventory')
        bom_data = data_loader.data.get('style_bom')
        
        if knit_orders is None or knit_orders.empty:
            return {"status": "error", "message": "No knit orders data available"}
        
        suggestions = []
        
        # Analyze each order for production suggestions
        for idx, order in knit_orders.iterrows():
            style = order.get('Style #', order.get('Style', 'Unknown'))
            quantity_ordered = float(order.get('Qty Ordered (lbs)', 0))
            balance = float(order.get('Balance (lbs)', 0))
            shipped = float(order.get('Shipped (lbs)', 0))
            g00_stage = float(order.get('G00 (lbs)', 0)) if pd.notna(order.get('G00 (lbs)')) else 0
            
            # Skip completed orders
            if balance <= 0:
                continue
            
            # Calculate production priority based on multiple factors
            completion_pct = (shipped / quantity_ordered * 100) if quantity_ordered > 0 else 0
            
            # Determine priority
            if completion_pct < 25:
                priority = "HIGH"
                priority_score = 1
            elif completion_pct < 50:
                priority = "MEDIUM"
                priority_score = 2
            else:
                priority = "LOW"
                priority_score = 3
            
            # Check material availability if BOM data exists
            material_status = "Unknown"
            material_available = 0
            material_required = balance  # Simplified: 1:1 ratio
            
            if yarn_inventory is not None and not yarn_inventory.empty:
                # Calculate total available yarn
                total_yarn = yarn_inventory['Physical Inventory'].sum() if 'Physical Inventory' in yarn_inventory.columns else 0
                material_available = total_yarn / max(1, len(knit_orders))  # Rough allocation
                
                if material_available >= material_required:
                    material_status = f"Available: {material_available:.0f} lbs"
                else:
                    shortage = material_required - material_available
                    material_status = f"Shortage: {shortage:.0f} lbs (need {material_required:.0f} lbs, have {material_available:.0f} lbs)"
            
            # Calculate suggested production quantity
            # Consider current stage and balance
            if g00_stage > 0:
                # Already in production, suggest completing the balance
                suggested_qty = balance
                reason = "Complete in-progress order"
            else:
                # Not started, suggest based on priority
                if priority == "HIGH":
                    suggested_qty = min(balance, 5000)  # Max batch size
                    reason = "High priority order - start production"
                elif priority == "MEDIUM":
                    suggested_qty = min(balance, 3000)
                    reason = "Medium priority - schedule production"
                else:
                    suggested_qty = min(balance, 1000)
                    reason = "Low priority - batch with similar styles"
            
            # Calculate lead time based on quantity
            lead_time_days = max(3, int(suggested_qty / 1000) * 2)  # 2 days per 1000 lbs
            
            # Calculate confidence score
            confidence = 0.9 if material_available >= material_required else 0.6
            if g00_stage > 0:
                confidence += 0.1  # Higher confidence for in-progress orders
            
            suggestion = {
                'style': style,
                'order_number': order.get('Order #', f'KO-{idx+1:04d}'),
                'current_balance': balance,
                'suggested_quantity': suggested_qty,
                'priority': priority,
                'priority_score': priority_score,
                'material_status': material_status,
                'material_available': material_available,
                'material_required': material_required,
                'reason': reason,
                'lead_time_days': lead_time_days,
                'confidence': min(1.0, confidence),
                'production_stage': 'Greige' if g00_stage > 0 else 'Not Started',
                'completion_percentage': round(completion_pct, 1)
            }
            
            suggestions.append(suggestion)
        
        # Sort by priority score and confidence
        suggestions.sort(key=lambda x: (x['priority_score'], -x['confidence']))
        
        # Take top 10 suggestions
        top_suggestions = suggestions[:10]
        
        # Calculate summary
        total_suggested = sum(s['suggested_quantity'] for s in top_suggestions)
        material_available_count = len([s for s in top_suggestions if 'Available' in s['material_status']])
        material_shortage_count = len([s for s in top_suggestions if 'Shortage' in s['material_status']])
        
        summary = {
            'total_suggestions': len(top_suggestions),
            'total_suggested_production': total_suggested,
            'material_available': material_available_count,
            'material_shortage': material_shortage_count,
            'high_priority': len([s for s in top_suggestions if s['priority'] == 'HIGH']),
            'medium_priority': len([s for s in top_suggestions if s['priority'] == 'MEDIUM']),
            'low_priority': len([s for s in top_suggestions if s['priority'] == 'LOW'])
        }
        
        return {
            'status': 'success',
            'suggestions': top_suggestions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'suggestions': [],
            'summary': {}
        }