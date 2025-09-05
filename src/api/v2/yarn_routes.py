"""
Yarn API Routes v2
Handles all yarn-related endpoints including intelligence and substitution
"""

from flask import Blueprint, jsonify, request
from src.services.service_container import get_service
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Create blueprint
yarn_bp = Blueprint('yarn_v2', __name__)


@yarn_bp.route('/yarn/intelligence', methods=['GET'])
def get_yarn_intelligence():
    """Get yarn intelligence analysis"""
    try:
        yarn_service = get_service('yarn')
        inventory_service = get_service('inventory')
        
        # Get parameters
        analysis = request.args.get('analysis', 'all')
        forecast = request.args.get('forecast', 'false').lower() == 'true'
        
        intelligence_data = {
            'total_yarns': 0,
            'shortage_analysis': {},
            'substitution_options': [],
            'inventory_health': {},
            'recommendations': []
        }
        
        # Get yarn inventory
        yarn_data = inventory_service.get_yarn_inventory()
        
        if yarn_data is not None:
            if isinstance(yarn_data, pd.DataFrame):
                intelligence_data['total_yarns'] = len(yarn_data)
            
            # Shortage analysis
            if analysis in ['all', 'shortage']:
                shortages = inventory_service.detect_shortages()
                if isinstance(shortages, pd.DataFrame):
                    intelligence_data['shortage_analysis'] = {
                        'count': len(shortages),
                        'critical': len(shortages[shortages['planning_balance'] < -1000]),
                        'items': shortages.head(20).to_dict('records')
                    }
            
            # Get substitution options for shortages
            if isinstance(shortages, pd.DataFrame) and not shortages.empty:
                for _, yarn in shortages.head(5).iterrows():
                    subs = yarn_service.find_substitutions(yarn.get('yarn_id'))
                    if subs:
                        intelligence_data['substitution_options'].extend(subs)
        
        # Add forecast if requested
        if forecast:
            forecasting_service = get_service('forecasting')
            yarn_forecast = forecasting_service.forecast_yarn_demand()
            intelligence_data['forecast'] = yarn_forecast
        
        # Generate recommendations
        if intelligence_data['shortage_analysis'].get('critical', 0) > 5:
            intelligence_data['recommendations'].append({
                'type': 'urgent',
                'action': 'Expedite orders for critical shortages',
                'count': intelligence_data['shortage_analysis']['critical']
            })
        
        return jsonify({
            'status': 'success',
            'intelligence': intelligence_data
        })
        
    except Exception as e:
        logger.error(f"Error in yarn intelligence: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/requirements', methods=['POST'])
def calculate_yarn_requirements():
    """Calculate yarn requirements for production"""
    try:
        yarn_service = get_service('yarn')
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        style_id = data.get('style_id')
        quantity = data.get('quantity')
        
        if not style_id or quantity is None:
            return jsonify({
                'status': 'error',
                'message': 'style_id and quantity required'
            }), 400
        
        # Calculate requirements
        requirements = yarn_service.calculate_requirements(style_id, quantity)
        
        return jsonify({
            'status': 'success',
            'style_id': style_id,
            'quantity': quantity,
            'requirements': requirements
        })
        
    except Exception as e:
        logger.error(f"Error calculating requirements: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/substitution', methods=['GET'])
def get_yarn_substitutions():
    """Get intelligent yarn substitution suggestions"""
    try:
        yarn_service = get_service('yarn')
        
        yarn_id = request.args.get('yarn_id')
        
        if not yarn_id:
            return jsonify({
                'status': 'error',
                'message': 'yarn_id required'
            }), 400
        
        # Find substitutions
        substitutions = yarn_service.find_substitutions(yarn_id)
        
        # Rank by suitability
        ranked_subs = yarn_service.rank_substitutions(substitutions)
        
        return jsonify({
            'status': 'success',
            'yarn_id': yarn_id,
            'substitutions': ranked_subs,
            'count': len(ranked_subs)
        })
        
    except Exception as e:
        logger.error(f"Error finding substitutions: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/availability-check', methods=['POST'])
def check_yarn_availability():
    """Check yarn availability for production order"""
    try:
        yarn_service = get_service('yarn')
        inventory_service = get_service('inventory')
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        style_id = data.get('style_id')
        quantity = data.get('quantity')
        
        if not style_id or quantity is None:
            return jsonify({
                'status': 'error',
                'message': 'style_id and quantity required'
            }), 400
        
        # Calculate requirements
        requirements = yarn_service.calculate_requirements(style_id, quantity)
        
        # Check availability
        availability = {
            'available': True,
            'shortages': [],
            'substitutions_needed': []
        }
        
        for yarn_id, required_qty in requirements.items():
            # Get current balance
            balance = inventory_service.calculate_planning_balance(yarn_id)
            
            if balance < required_qty:
                shortage = {
                    'yarn_id': yarn_id,
                    'required': required_qty,
                    'available': max(0, balance),
                    'shortage': required_qty - max(0, balance)
                }
                availability['shortages'].append(shortage)
                availability['available'] = False
                
                # Find substitutions
                subs = yarn_service.find_substitutions(yarn_id)
                if subs:
                    availability['substitutions_needed'].append({
                        'yarn_id': yarn_id,
                        'alternatives': subs[:3]  # Top 3 alternatives
                    })
        
        return jsonify({
            'status': 'success',
            'availability': availability
        })
        
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/interchangeability', methods=['GET'])
def get_yarn_interchangeability():
    """Get yarn interchangeability analysis"""
    try:
        yarn_service = get_service('yarn')
        
        yarn_id = request.args.get('yarn_id')
        category = request.args.get('category')
        
        if yarn_id:
            # Get interchangeable yarns for specific yarn
            interchangeable = yarn_service.get_interchangeable_yarns(yarn_id)
        elif category:
            # Get all interchangeable yarns in category
            interchangeable = yarn_service.get_category_interchangeability(category)
        else:
            # Get general interchangeability matrix
            interchangeable = yarn_service.get_interchangeability_matrix()
        
        return jsonify({
            'status': 'success',
            'interchangeability': interchangeable
        })
        
    except Exception as e:
        logger.error(f"Error getting interchangeability: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/consumption', methods=['GET'])
def get_yarn_consumption():
    """Get yarn consumption analytics"""
    try:
        yarn_service = get_service('yarn')
        
        # Get time period
        period = request.args.get('period', '30')  # days
        yarn_id = request.args.get('yarn_id')
        
        consumption_data = {
            'period_days': int(period),
            'total_consumption': 0,
            'by_yarn': {},
            'by_style': {},
            'trends': []
        }
        
        # Get consumption data
        consumption = yarn_service.analyze_consumption(int(period), yarn_id)
        
        if consumption:
            consumption_data.update(consumption)
        
        return jsonify({
            'status': 'success',
            'consumption': consumption_data
        })
        
    except Exception as e:
        logger.error(f"Error getting consumption: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/forecast', methods=['GET'])
def get_yarn_forecast():
    """Get yarn demand forecast"""
    try:
        forecasting_service = get_service('forecasting')
        
        # Get parameters
        horizon = int(request.args.get('horizon', 30))
        yarn_id = request.args.get('yarn_id')
        confidence = float(request.args.get('confidence', 0.95))
        
        if yarn_id:
            # Forecast specific yarn
            forecast = forecasting_service.forecast_yarn_demand(
                yarn_id=yarn_id,
                horizon=horizon,
                confidence=confidence
            )
        else:
            # Forecast all yarns
            forecast = forecasting_service.forecast_all_yarns(
                horizon=horizon,
                confidence=confidence
            )
        
        return jsonify({
            'status': 'success',
            'forecast': forecast,
            'horizon_days': horizon,
            'confidence_level': confidence
        })
        
    except Exception as e:
        logger.error(f"Error getting yarn forecast: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@yarn_bp.route('/yarn/safety-stock', methods=['GET', 'POST'])
def handle_safety_stock():
    """Get or update safety stock levels"""
    try:
        yarn_service = get_service('yarn')
        
        if request.method == 'GET':
            yarn_id = request.args.get('yarn_id')
            
            if yarn_id:
                safety_stock = yarn_service.get_safety_stock(yarn_id)
            else:
                safety_stock = yarn_service.get_all_safety_stocks()
            
            return jsonify({
                'status': 'success',
                'safety_stock': safety_stock
            })
            
        elif request.method == 'POST':
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400
            
            yarn_id = data.get('yarn_id')
            new_level = data.get('safety_stock_level')
            
            if not yarn_id or new_level is None:
                return jsonify({
                    'status': 'error',
                    'message': 'yarn_id and safety_stock_level required'
                }), 400
            
            # Update safety stock
            success = yarn_service.update_safety_stock(yarn_id, new_level)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Safety stock updated',
                    'yarn_id': yarn_id,
                    'new_level': new_level
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to update safety stock'
                }), 500
                
    except Exception as e:
        logger.error(f"Error handling safety stock: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500