"""
Inventory API Routes v2
Handles all inventory-related endpoints
"""

from flask import Blueprint, jsonify, request
from src.services.service_container import get_service
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Create blueprint
inventory_bp = Blueprint('inventory_v2', __name__)


@inventory_bp.route('/inventory', methods=['GET'])
def get_inventory():
    """Get inventory intelligence with enhanced features"""
    try:
        # Get service
        inventory_service = get_service('inventory')
        
        # Get query parameters
        view = request.args.get('view', 'summary')
        realtime = request.args.get('realtime', 'false').lower() == 'true'
        analysis = request.args.get('analysis', 'all')
        
        # Get inventory data based on view
        if view == 'summary':
            data = inventory_service.get_summary()
        elif view == 'detailed':
            data = inventory_service.get_detailed_inventory()
        elif view == 'shortages':
            data = inventory_service.detect_shortages()
            # Convert DataFrame to dict if necessary
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')
        elif view == 'yarn':
            data = inventory_service.get_yarn_inventory()
        else:
            data = inventory_service.get_enhanced_intelligence()
        
        # Add analysis if requested
        if analysis == 'shortage':
            shortages = inventory_service.detect_shortages()
            if isinstance(shortages, pd.DataFrame):
                data['shortage_analysis'] = shortages.to_dict('records')
            else:
                data['shortage_analysis'] = shortages
        
        return jsonify({
            'status': 'success',
            'data': data,
            'view': view,
            'realtime': realtime
        })
        
    except Exception as e:
        logger.error(f"Error in get_inventory: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@inventory_bp.route('/inventory/shortages', methods=['GET'])
def get_shortages():
    """Get yarn shortages"""
    try:
        inventory_service = get_service('inventory')
        
        # Get parameters
        threshold = float(request.args.get('threshold', 0))
        critical_only = request.args.get('critical', 'false').lower() == 'true'
        
        # Get shortages
        shortages = inventory_service.detect_shortages(threshold=threshold)
        
        # Filter for critical if requested
        if critical_only and isinstance(shortages, pd.DataFrame):
            shortages = shortages[shortages['planning_balance'] < -1000]
        
        # Convert to dict
        if isinstance(shortages, pd.DataFrame):
            result = shortages.to_dict('records')
        else:
            result = shortages
        
        return jsonify({
            'status': 'success',
            'shortages': result,
            'count': len(result),
            'threshold': threshold
        })
        
    except Exception as e:
        logger.error(f"Error getting shortages: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@inventory_bp.route('/inventory/planning-balance', methods=['GET'])
def get_planning_balance():
    """Get planning balance for specific yarn or all yarns"""
    try:
        inventory_service = get_service('inventory')
        
        yarn_id = request.args.get('yarn_id')
        
        if yarn_id:
            # Get specific yarn balance
            balance = inventory_service.calculate_planning_balance(yarn_id)
            return jsonify({
                'status': 'success',
                'yarn_id': yarn_id,
                'planning_balance': balance
            })
        else:
            # Get all balances
            inventory_data = inventory_service.get_detailed_inventory()
            
            if isinstance(inventory_data, pd.DataFrame):
                balances = inventory_data[['yarn_id', 'planning_balance']].to_dict('records')
            else:
                balances = []
            
            return jsonify({
                'status': 'success',
                'balances': balances,
                'count': len(balances)
            })
            
    except Exception as e:
        logger.error(f"Error getting planning balance: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@inventory_bp.route('/inventory/update', methods=['POST'])
def update_inventory():
    """Update inventory levels"""
    try:
        inventory_service = get_service('inventory')
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        yarn_id = data.get('yarn_id')
        new_balance = data.get('balance')
        
        if not yarn_id or new_balance is None:
            return jsonify({
                'status': 'error',
                'message': 'yarn_id and balance required'
            }), 400
        
        # Update balance
        success = inventory_service.update_balance(yarn_id, new_balance)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Inventory updated',
                'yarn_id': yarn_id,
                'new_balance': new_balance
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update inventory'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating inventory: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@inventory_bp.route('/inventory/netting', methods=['GET'])
def get_inventory_netting():
    """Get multi-level inventory netting calculations"""
    try:
        inventory_service = get_service('inventory')
        mrp_service = get_service('mrp')
        
        # Perform netting calculations
        netting_data = {
            'gross_requirements': {},
            'scheduled_receipts': {},
            'on_hand': {},
            'net_requirements': {},
            'planned_orders': {}
        }
        
        # Get current inventory
        current_inventory = inventory_service.get_detailed_inventory()
        
        if isinstance(current_inventory, pd.DataFrame):
            for _, row in current_inventory.iterrows():
                yarn_id = row.get('yarn_id')
                netting_data['on_hand'][yarn_id] = row.get('planning_balance', 0)
        
        # Add MRP calculations if available
        try:
            demand_forecast = pd.DataFrame()  # Would get from forecasting service
            mrp_result = mrp_service.calculate_requirements(demand_forecast)
            
            if 'mrp_table' in mrp_result:
                netting_data.update(mrp_result)
        except:
            pass
        
        return jsonify({
            'status': 'success',
            'netting': netting_data
        })
        
    except Exception as e:
        logger.error(f"Error calculating netting: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@inventory_bp.route('/inventory/analytics', methods=['GET'])
def get_inventory_analytics():
    """Get inventory analytics and insights"""
    try:
        inventory_service = get_service('inventory')
        
        analytics = {
            'turnover_rate': 4.5,  # Placeholder
            'carrying_cost': 50000,
            'stockout_risk': [],
            'excess_inventory': [],
            'abc_classification': {},
            'velocity_analysis': {}
        }
        
        # Get shortage analysis
        shortages = inventory_service.detect_shortages()
        if isinstance(shortages, pd.DataFrame):
            analytics['stockout_risk'] = shortages.head(10).to_dict('records')
        
        return jsonify({
            'status': 'success',
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500