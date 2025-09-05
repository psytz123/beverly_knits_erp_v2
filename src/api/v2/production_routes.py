"""
Production API Routes v2
Handles all production and scheduling endpoints
"""

from flask import Blueprint, jsonify, request
from src.services.service_container import get_service
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
production_bp = Blueprint('production_v2', __name__)


@production_bp.route('/production/planning', methods=['GET'])
def get_production_planning():
    """Get production planning data"""
    try:
        scheduler_service = get_service('scheduler')
        capacity_service = get_service('capacity')
        
        # Get parameters
        view = request.args.get('view', 'orders')
        forecast = request.args.get('forecast', 'false').lower() == 'true'
        horizon_days = int(request.args.get('horizon', 30))
        
        # Get production orders
        data_loader = get_service('data_loader')
        orders = data_loader.load_production_orders()
        
        planning_data = {
            'orders': [],
            'capacity': {},
            'schedule': [],
            'metrics': {}
        }
        
        if orders is not None and not orders.empty:
            # Schedule orders
            scheduled_orders = scheduler_service.schedule_production(orders)
            
            if view == 'orders':
                planning_data['orders'] = scheduled_orders.to_dict('records') if isinstance(scheduled_orders, pd.DataFrame) else scheduled_orders
            
            # Get capacity analysis
            planning_data['capacity'] = capacity_service.get_capacity_analysis()
            
            # Get metrics
            planning_data['metrics'] = scheduler_service.get_schedule_metrics(scheduled_orders)
        
        # Add forecast if requested
        if forecast:
            forecasting_service = get_service('forecasting')
            forecast_data = forecasting_service.generate_comprehensive_forecast(horizon=horizon_days)
            planning_data['forecast'] = forecast_data
        
        return jsonify({
            'status': 'success',
            'data': planning_data,
            'view': view
        })
        
    except Exception as e:
        logger.error(f"Error in production planning: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/status', methods=['GET'])
def get_production_status():
    """Get current production status"""
    try:
        scheduler_service = get_service('scheduler')
        data_loader = get_service('data_loader')
        
        # Load production orders
        orders = data_loader.load_production_orders()
        
        status = {
            'total_orders': 0,
            'in_progress': 0,
            'scheduled': 0,
            'completed': 0,
            'pending': 0,
            'machine_utilization': {}
        }
        
        if orders is not None and not orders.empty:
            status['total_orders'] = len(orders)
            
            # Count by status
            if 'status' in orders.columns:
                status_counts = orders['status'].value_counts().to_dict()
                status['in_progress'] = status_counts.get('in_progress', 0)
                status['scheduled'] = status_counts.get('scheduled', 0)
                status['completed'] = status_counts.get('completed', 0)
                status['pending'] = status_counts.get('pending', 0)
            
            # Get machine utilization
            status['machine_utilization'] = scheduler_service.get_machine_utilization()
        
        return jsonify({
            'status': 'success',
            'production_status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting production status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/capacity', methods=['GET'])
def get_production_capacity():
    """Get production capacity analysis"""
    try:
        capacity_service = get_service('capacity')
        
        capacity_data = capacity_service.get_capacity_analysis()
        
        return jsonify({
            'status': 'success',
            'capacity': capacity_data
        })
        
    except Exception as e:
        logger.error(f"Error getting capacity: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/orders', methods=['GET', 'POST'])
def handle_production_orders():
    """Get or create production orders"""
    try:
        if request.method == 'GET':
            # Get orders with filters
            status = request.args.get('status')
            machine_id = request.args.get('machine_id')
            style_id = request.args.get('style_id')
            
            data_loader = get_service('data_loader')
            orders = data_loader.load_production_orders()
            
            if orders is not None and not orders.empty:
                # Apply filters
                if status:
                    orders = orders[orders['status'] == status]
                if machine_id:
                    orders = orders[orders['machine_id'] == machine_id]
                if style_id:
                    orders = orders[orders['style_id'] == style_id]
                
                result = orders.to_dict('records')
            else:
                result = []
            
            return jsonify({
                'status': 'success',
                'orders': result,
                'count': len(result)
            })
            
        elif request.method == 'POST':
            # Create new production order
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400
            
            # Validate required fields
            required = ['style_id', 'quantity']
            for field in required:
                if field not in data:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing required field: {field}'
                    }), 400
            
            # Create order
            order = {
                'order_id': f"PO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'style_id': data['style_id'],
                'quantity': data['quantity'],
                'priority': data.get('priority', 'normal'),
                'status': 'pending',
                'created_date': datetime.now().isoformat()
            }
            
            # Schedule the order
            scheduler_service = get_service('scheduler')
            scheduled = scheduler_service.schedule_production(pd.DataFrame([order]))
            
            if isinstance(scheduled, pd.DataFrame) and not scheduled.empty:
                order = scheduled.iloc[0].to_dict()
            
            return jsonify({
                'status': 'success',
                'order': order,
                'message': 'Production order created'
            }), 201
            
    except Exception as e:
        logger.error(f"Error handling production orders: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/orders/<order_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_specific_order(order_id):
    """Handle specific production order"""
    try:
        data_loader = get_service('data_loader')
        orders = data_loader.load_production_orders()
        
        if orders is None or orders.empty:
            return jsonify({
                'status': 'error',
                'message': 'No orders found'
            }), 404
        
        # Find specific order
        order_data = orders[orders['order_id'] == order_id]
        
        if order_data.empty:
            return jsonify({
                'status': 'error',
                'message': f'Order {order_id} not found'
            }), 404
        
        if request.method == 'GET':
            return jsonify({
                'status': 'success',
                'order': order_data.iloc[0].to_dict()
            })
            
        elif request.method == 'PUT':
            # Update order
            update_data = request.get_json()
            
            # Update fields
            for key, value in update_data.items():
                orders.loc[orders['order_id'] == order_id, key] = value
            
            # Save updates (in production, this would update database)
            
            return jsonify({
                'status': 'success',
                'message': 'Order updated',
                'order_id': order_id
            })
            
        elif request.method == 'DELETE':
            # Delete order (mark as cancelled)
            orders.loc[orders['order_id'] == order_id, 'status'] = 'cancelled'
            
            return jsonify({
                'status': 'success',
                'message': 'Order cancelled',
                'order_id': order_id
            })
            
    except Exception as e:
        logger.error(f"Error handling order {order_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/assign-machine', methods=['POST'])
def assign_machine():
    """Assign machine to production order"""
    try:
        data = request.get_json()
        
        if not data or 'order_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'order_id required'
            }), 400
        
        order_id = data['order_id']
        
        # Get scheduler service
        scheduler_service = get_service('scheduler')
        data_loader = get_service('data_loader')
        
        # Load orders
        orders = data_loader.load_production_orders()
        
        if orders is None or orders.empty:
            return jsonify({
                'status': 'error',
                'message': 'No orders found'
            }), 404
        
        # Find order
        order_idx = orders[orders['order_id'] == order_id].index
        
        if len(order_idx) == 0:
            return jsonify({
                'status': 'error',
                'message': f'Order {order_id} not found'
            }), 404
        
        # Schedule just this order
        order_df = orders.loc[order_idx]
        scheduled = scheduler_service.schedule_production(order_df)
        
        if isinstance(scheduled, pd.DataFrame) and not scheduled.empty:
            machine_id = scheduled.iloc[0].get('machine_id')
            
            if machine_id:
                return jsonify({
                    'status': 'success',
                    'order_id': order_id,
                    'machine_id': machine_id,
                    'message': 'Machine assigned'
                })
        
        return jsonify({
            'status': 'error',
            'message': 'Could not assign machine'
        }), 500
        
    except Exception as e:
        logger.error(f"Error assigning machine: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/schedule/optimize', methods=['POST'])
def optimize_schedule():
    """Optimize production schedule"""
    try:
        scheduler_service = get_service('scheduler')
        data_loader = get_service('data_loader')
        
        # Get optimization parameters
        data = request.get_json() or {}
        goals = data.get('goals', {
            'minimize_changeover': True,
            'balance_workload': True,
            'prioritize_urgent': True
        })
        
        # Load current schedule
        orders = data_loader.load_production_orders()
        
        if orders is None or orders.empty:
            return jsonify({
                'status': 'error',
                'message': 'No orders to optimize'
            }), 404
        
        # Optimize
        optimized = scheduler_service.optimize_schedule(orders, goals)
        
        # Calculate improvements
        original_metrics = scheduler_service.get_schedule_metrics(orders)
        optimized_metrics = scheduler_service.get_schedule_metrics(optimized)
        
        return jsonify({
            'status': 'success',
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'improvements': {
                'lead_time_reduction': original_metrics.get('average_lead_time', 0) - optimized_metrics.get('average_lead_time', 0),
                'utilization_improvement': optimized_metrics.get('utilization', {})
            }
        })
        
    except Exception as e:
        logger.error(f"Error optimizing schedule: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@production_bp.route('/production/recommendations', methods=['GET'])
def get_production_recommendations():
    """Get AI-powered production recommendations"""
    try:
        scheduler_service = get_service('scheduler')
        capacity_service = get_service('capacity')
        
        recommendations = []
        
        # Check capacity utilization
        capacity = capacity_service.get_capacity_analysis()
        utilization = capacity.get('utilization_percentage', 0)
        
        if utilization > 90:
            recommendations.append({
                'type': 'capacity',
                'priority': 'high',
                'recommendation': 'Consider adding shifts or overtime',
                'reason': f'Capacity utilization at {utilization:.1f}%'
            })
        elif utilization < 60:
            recommendations.append({
                'type': 'capacity',
                'priority': 'medium',
                'recommendation': 'Consolidate production runs',
                'reason': f'Low capacity utilization at {utilization:.1f}%'
            })
        
        # Check for unassigned orders
        data_loader = get_service('data_loader')
        orders = data_loader.load_production_orders()
        
        if orders is not None and not orders.empty:
            unassigned = orders[orders['machine_id'].isna()]
            if len(unassigned) > 0:
                recommendations.append({
                    'type': 'scheduling',
                    'priority': 'high',
                    'recommendation': f'Assign {len(unassigned)} pending orders to machines',
                    'reason': 'Orders waiting for machine assignment'
                })
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500