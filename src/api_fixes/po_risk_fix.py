"""
Fixed PO Risk Analysis API - Uses real knit orders data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from src.utils.json_sanitizer import sanitize_for_json, safe_float

def get_po_risk_analysis_fixed(data_loader) -> Dict[str, Any]:
    """
    Generate PO risk analysis from real knit orders data
    """
    try:
        # Get real knit orders data
        if not data_loader or not hasattr(data_loader, 'data'):
            return {"status": "error", "message": "Data loader not available"}
        
        knit_orders = data_loader.data.get('knit_orders')
        if knit_orders is None or knit_orders.empty:
            return {"status": "error", "message": "No knit orders data available"}
        
        # Process each order for risk analysis
        risk_analysis = []
        current_date = datetime.now()
        
        for idx, row in knit_orders.iterrows():
            # Extract order data
            order_info = {
                'po_number': row.get('Order #', f'KO-{idx+1:04d}'),
                'style': row.get('Style #', row.get('Style', 'Unknown')),
                'customer': row.get('Customer', 'Beverly Knits'),  # Add customer mapping if available
                'quantity': safe_float(row.get('Qty Ordered (lbs)', 0)),
                'balance': safe_float(row.get('Balance (lbs)', 0)),
                'shipped': safe_float(row.get('Shipped (lbs)', 0)),
                'g00_stage': safe_float(row.get('G00 (lbs)', 0)) if pd.notna(row.get('G00 (lbs)')) else 0,
            }
            
            # Parse dates
            start_date = pd.to_datetime(row.get('Start Date'), errors='coerce')
            quoted_date = pd.to_datetime(row.get('Quoted Date'), errors='coerce')
            
            # Calculate due date (use quoted date or start date + 30 days)
            if pd.notna(quoted_date):
                due_date = quoted_date
            elif pd.notna(start_date):
                due_date = start_date + timedelta(days=30)
            else:
                due_date = current_date + timedelta(days=14)  # Default 2 weeks
            
            order_info['due_date'] = due_date.isoformat() if pd.notna(due_date) else None
            order_info['start_date'] = start_date.isoformat() if pd.notna(start_date) else None
            
            # Calculate days until due
            if pd.notna(due_date):
                days_until_due = (due_date - current_date).days
                order_info['days_until_due'] = days_until_due
            else:
                days_until_due = 999
                order_info['days_until_due'] = None
            
            # Calculate completion percentage
            if order_info['quantity'] > 0:
                completion_pct = (order_info['shipped'] / order_info['quantity']) * 100
            else:
                completion_pct = 0
            order_info['completion_percentage'] = round(completion_pct, 1)
            
            # Determine status
            if completion_pct >= 100:
                status = 'Complete'
            elif completion_pct > 0:
                status = 'In Progress'
            elif order_info['g00_stage'] > 0:
                status = 'In Production'
            else:
                status = 'Not Started'
            order_info['status'] = status
            
            # Calculate risk level
            if status == 'Complete':
                risk_level = 'LOW'
            elif days_until_due < 0:
                risk_level = 'OVERDUE'
            elif days_until_due <= 3:
                risk_level = 'CRITICAL'
            elif days_until_due <= 7:
                risk_level = 'HIGH'
            elif days_until_due <= 14:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            order_info['risk_level'] = risk_level
            
            # Add production metrics
            order_info['production_stage'] = 'Greige' if order_info['g00_stage'] > 0 else 'Knitting'
            order_info['remaining_production'] = order_info['balance']
            
            risk_analysis.append(order_info)
        
        # Sort by risk level and days until due
        risk_priority = {'OVERDUE': 0, 'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
        risk_analysis.sort(key=lambda x: (risk_priority.get(x['risk_level'], 5), x.get('days_until_due', 999)))
        
        # Get complete orders (top 5)
        complete_orders = [o for o in risk_analysis if o['status'] == 'Complete'][:5]
        
        # Calculate summary statistics
        summary = {
            'total_orders': len(risk_analysis),
            'overdue_orders': len([o for o in risk_analysis if o['risk_level'] == 'OVERDUE']),
            'critical_orders': len([o for o in risk_analysis if o['risk_level'] == 'CRITICAL']),
            'high_risk_orders': len([o for o in risk_analysis if o['risk_level'] == 'HIGH']),
            'medium_risk_orders': len([o for o in risk_analysis if o['risk_level'] == 'MEDIUM']),
            'low_risk_orders': len([o for o in risk_analysis if o['risk_level'] == 'LOW']),
            'complete_orders': len([o for o in risk_analysis if o['status'] == 'Complete']),
            'in_progress_orders': len([o for o in risk_analysis if o['status'] == 'In Progress']),
            'not_started_orders': len([o for o in risk_analysis if o['status'] == 'Not Started']),
            'total_quantity_lbs': sum(o['quantity'] for o in risk_analysis),
            'total_shipped_lbs': sum(o['shipped'] for o in risk_analysis),
            'total_balance_lbs': sum(o['balance'] for o in risk_analysis),
        }
        
        result = {
            'status': 'success',
            'risk_analysis': risk_analysis[:50],  # Return top 50 for display
            'complete_orders': complete_orders,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sanitize the entire result to remove NaN values
        return sanitize_for_json(result)
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'risk_analysis': [],
            'complete_orders': [],
            'summary': {}
        }