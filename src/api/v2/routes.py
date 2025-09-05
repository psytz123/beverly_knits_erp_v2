"""Consolidated API v2 routes using dependency injection."""

from flask import Blueprint, request, jsonify, Response
import pandas as pd
from typing import Dict, Any
import json
from datetime import datetime
import logging

from src.infrastructure.container.flask_integration import inject_service, get_service
from src.application.services.inventory_service import InventoryService
from src.infrastructure.repositories.yarn_repository import YarnRepository


# Create v2 API blueprint
api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')
logger = logging.getLogger(__name__)


# ============================================================================
# INVENTORY ENDPOINTS (Consolidates 5 endpoints)
# ============================================================================

@api_v2.route('/inventory', methods=['GET'])
@inject_service('inventory_analyzer')
@inject_service('cache_manager')
def inventory_endpoint(inventory_analyzer=None, cache_manager=None):
    """
    Consolidated inventory endpoint.
    
    Replaces:
    - /api/yarn-inventory
    - /api/yarn-data
    - /api/inventory-intelligence-enhanced
    - /api/real-time-inventory-dashboard
    - /api/emergency-shortage-dashboard
    
    Query Parameters:
    - view: summary|detailed|yarn|shortage|analytics
    - analysis: none|shortage|forecast|intelligence
    - realtime: true|false
    - format: json|csv|excel
    - threshold: float (for shortage detection)
    - limit: int (pagination)
    - offset: int (pagination)
    """
    try:
        # Parse query parameters
        view = request.args.get('view', 'summary')
        analysis = request.args.get('analysis', 'none')
        realtime = request.args.get('realtime', 'false').lower() == 'true'
        output_format = request.args.get('format', 'json')
        threshold = float(request.args.get('threshold', 0))
        limit = int(request.args.get('limit', 1000))
        offset = int(request.args.get('offset', 0))
        
        # Get data based on view
        if view == 'summary':
            data = inventory_analyzer.get_inventory_summary()
        elif view == 'detailed':
            data = inventory_analyzer.get_detailed_inventory()
        elif view == 'yarn':
            data = inventory_analyzer.get_yarn_inventory()
        elif view == 'shortage':
            data = inventory_analyzer.detect_shortages(threshold=threshold)
        elif view == 'analytics':
            data = inventory_analyzer.get_inventory_analytics()
        else:
            data = inventory_analyzer.get_inventory_summary()
        
        # Apply analysis if requested
        if analysis == 'shortage':
            shortages = inventory_analyzer.detect_shortages(threshold=threshold)
            data = {
                'inventory': data,
                'shortage_analysis': shortages,
                'critical_count': len([s for s in shortages if s.get('severity') == 'critical'])
            }
        elif analysis == 'intelligence':
            intelligence = inventory_analyzer.get_yarn_intelligence()
            data = {
                'inventory': data,
                'intelligence': intelligence
            }
        elif analysis == 'forecast' and 'sales_forecasting' in request.app.config.get('DI_CONTAINER', {}).__dict__:
            forecasting = get_service('sales_forecasting')
            forecast_data = forecasting.generate_forecast(horizon_days=30)
            data = {
                'inventory': data,
                'forecast': forecast_data
            }
        
        # Format output
        if output_format == 'csv' and isinstance(data, (list, dict)):
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            return Response(
                df.to_csv(index=False),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment;filename=inventory.csv'}
            )
        elif output_format == 'excel' and isinstance(data, (list, dict)):
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            output = df.to_excel(index=False, engine='openpyxl')
            return Response(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': 'attachment;filename=inventory.xlsx'}
            )
        else:
            return jsonify({
                'status': 'success',
                'data': data,
                'metadata': {
                    'view': view,
                    'analysis': analysis,
                    'realtime': realtime,
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
    except Exception as e:
        logger.error(f"Error in inventory endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# PRODUCTION ENDPOINTS (Consolidates 6 endpoints)
# ============================================================================

@api_v2.route('/production', methods=['GET', 'POST'])
@inject_service('production_pipeline')
@inject_service('six_phase_planning')
def production_endpoint(production_pipeline=None, six_phase_planning=None):
    """
    Consolidated production endpoint.
    
    Replaces:
    - /api/production-planning
    - /api/production-status
    - /api/production-pipeline
    - /api/production-recommendations-ml
    - /api/machine-assignment-suggestions
    - /api/production-flow
    
    GET Parameters:
    - view: status|planning|pipeline|recommendations|machines|flow
    - style_id: string (filter by style)
    - machine_id: string (filter by machine)
    - status: pending|assigned|in_progress|completed
    
    POST: Create new production order
    """
    try:
        if request.method == 'GET':
            view = request.args.get('view', 'status')
            style_id = request.args.get('style_id')
            machine_id = request.args.get('machine_id')
            status_filter = request.args.get('status')
            
            if view == 'status':
                data = production_pipeline.get_production_status(
                    style_id=style_id,
                    machine_id=machine_id,
                    status=status_filter
                )
            elif view == 'planning':
                data = six_phase_planning.get_production_plan()
            elif view == 'pipeline':
                data = production_pipeline.get_pipeline_status()
            elif view == 'recommendations':
                suggestions = get_service('production_suggestions')
                data = suggestions.get_ml_recommendations()
            elif view == 'machines':
                data = production_pipeline.get_machine_assignments()
            elif view == 'flow':
                data = six_phase_planning.get_production_flow()
            else:
                data = production_pipeline.get_production_status()
            
            return jsonify({
                'status': 'success',
                'data': data,
                'metadata': {
                    'view': view,
                    'filters': {
                        'style_id': style_id,
                        'machine_id': machine_id,
                        'status': status_filter
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
        else:  # POST
            order_data = request.json
            
            # Validate required fields
            required = ['style_id', 'quantity', 'customer_name']
            missing = [f for f in required if f not in order_data]
            if missing:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required fields: {", ".join(missing)}'
                }), 400
            
            # Create production order
            result = production_pipeline.create_production_order(order_data)
            
            return jsonify({
                'status': 'success',
                'data': result,
                'message': 'Production order created successfully'
            }), 201
            
    except Exception as e:
        logger.error(f"Error in production endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# FORECASTING ENDPOINTS (Consolidates 4 endpoints)
# ============================================================================

@api_v2.route('/forecast', methods=['GET', 'POST'])
@inject_service('enhanced_forecasting')
@inject_service('forecast_monitor')
def forecast_endpoint(enhanced_forecasting=None, forecast_monitor=None):
    """
    Consolidated forecasting endpoint.
    
    Replaces:
    - /api/ml-forecasting
    - /api/ml-forecast-detailed
    - /api/forecast-accuracy
    - /api/demand-forecast
    
    GET Parameters:
    - style_id: string (specific style to forecast)
    - horizon: int (days to forecast, default 30)
    - model: arima|prophet|lstm|xgboost|ensemble
    - include_accuracy: true|false
    - detail: summary|full
    """
    try:
        if request.method == 'GET':
            style_id = request.args.get('style_id')
            horizon = int(request.args.get('horizon', 30))
            model = request.args.get('model', 'ensemble')
            include_accuracy = request.args.get('include_accuracy', 'false').lower() == 'true'
            detail = request.args.get('detail', 'summary')
            
            # Generate forecast
            forecast_data = enhanced_forecasting.generate_forecast(
                style_id=style_id,
                horizon_days=horizon,
                model_type=model
            )
            
            # Add accuracy metrics if requested
            if include_accuracy:
                accuracy_data = forecast_monitor.get_accuracy_metrics(
                    style_id=style_id,
                    model_type=model
                )
                forecast_data['accuracy_metrics'] = accuracy_data
            
            # Adjust detail level
            if detail == 'summary':
                # Simplify the response for summary view
                if isinstance(forecast_data, dict) and 'predictions' in forecast_data:
                    forecast_data = {
                        'summary': forecast_data.get('summary', {}),
                        'next_30_days': forecast_data.get('predictions', {}).get('30_days', []),
                        'confidence': forecast_data.get('confidence', 0.95)
                    }
            
            return jsonify({
                'status': 'success',
                'data': forecast_data,
                'metadata': {
                    'style_id': style_id,
                    'horizon_days': horizon,
                    'model': model,
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
        else:  # POST - Trigger retraining
            retrain_params = request.json or {}
            
            auto_retrain = get_service('forecast_auto_retrain')
            result = auto_retrain.trigger_retrain(
                model_type=retrain_params.get('model', 'all'),
                force=retrain_params.get('force', False)
            )
            
            return jsonify({
                'status': 'success',
                'data': result,
                'message': 'Forecast model retraining initiated'
            })
            
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# YARN INTELLIGENCE ENDPOINTS (Consolidates 3 endpoints)
# ============================================================================

@api_v2.route('/yarn', methods=['GET'])
@inject_service('yarn_intelligence')
@inject_service('intelligent_yarn_matcher')
def yarn_endpoint(yarn_intelligence=None, intelligent_yarn_matcher=None):
    """
    Consolidated yarn intelligence endpoint.
    
    Replaces:
    - /api/yarn-intelligence
    - /api/yarn-substitution-intelligent
    - /api/yarn-interchangeability
    
    Parameters:
    - action: status|substitution|interchangeability|analysis
    - yarn_id: string (specific yarn)
    - style_id: string (for substitution analysis)
    """
    try:
        action = request.args.get('action', 'status')
        yarn_id = request.args.get('yarn_id')
        style_id = request.args.get('style_id')
        
        if action == 'status':
            data = yarn_intelligence.get_yarn_status(yarn_id=yarn_id)
        elif action == 'substitution':
            if not yarn_id:
                return jsonify({
                    'status': 'error',
                    'message': 'yarn_id required for substitution analysis'
                }), 400
            
            data = intelligent_yarn_matcher.find_substitutes(
                yarn_id=yarn_id,
                style_id=style_id
            )
        elif action == 'interchangeability':
            interchangeability = get_service('yarn_interchangeability')
            data = interchangeability.analyze_interchangeability(
                yarn_id=yarn_id,
                style_id=style_id
            )
        elif action == 'analysis':
            data = yarn_intelligence.get_comprehensive_analysis(yarn_id=yarn_id)
        else:
            data = yarn_intelligence.get_yarn_status()
        
        return jsonify({
            'status': 'success',
            'data': data,
            'metadata': {
                'action': action,
                'yarn_id': yarn_id,
                'style_id': style_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in yarn endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# KPI ENDPOINTS (Consolidates 2 endpoints)
# ============================================================================

@api_v2.route('/kpis', methods=['GET'])
@inject_service('service_manager')
def kpi_endpoint(service_manager=None):
    """
    Consolidated KPI endpoint.
    
    Replaces:
    - /api/comprehensive-kpis
    - /api/key-metrics
    
    Parameters:
    - category: inventory|production|forecast|quality|all
    - period: daily|weekly|monthly
    """
    try:
        category = request.args.get('category', 'all')
        period = request.args.get('period', 'daily')
        
        # Get KPIs from service manager
        kpis = service_manager.get_comprehensive_kpis(
            category=category,
            period=period
        )
        
        return jsonify({
            'status': 'success',
            'data': kpis,
            'metadata': {
                'category': category,
                'period': period,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in KPI endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# NETTING ENDPOINTS (Consolidates 2 endpoints)
# ============================================================================

@api_v2.route('/netting', methods=['POST'])
@inject_service('inventory_analyzer')
@inject_service('business_rules')
def netting_endpoint(inventory_analyzer=None, business_rules=None):
    """
    Consolidated netting endpoint.
    
    Replaces:
    - /api/inventory-netting
    - /api/material-requirements
    
    POST Body:
    - style_id: string
    - quantity: float
    - include_forecast: boolean
    - check_substitutes: boolean
    """
    try:
        data = request.json or {}
        
        # Validate required fields
        if 'style_id' not in data or 'quantity' not in data:
            return jsonify({
                'status': 'error',
                'message': 'style_id and quantity are required'
            }), 400
        
        # Perform netting calculation
        netting_result = inventory_analyzer.perform_netting(
            style_id=data['style_id'],
            quantity=float(data['quantity']),
            include_forecast=data.get('include_forecast', False)
        )
        
        # Apply business rules
        validated_result = business_rules.validate_netting(netting_result)
        
        # Check for substitutes if requested
        if data.get('check_substitutes', False):
            yarn_matcher = get_service('intelligent_yarn_matcher')
            for shortage in validated_result.get('shortages', []):
                yarn_id = shortage.get('yarn_id')
                substitutes = yarn_matcher.find_substitutes(yarn_id)
                shortage['available_substitutes'] = substitutes
        
        return jsonify({
            'status': 'success',
            'data': validated_result,
            'metadata': {
                'style_id': data['style_id'],
                'quantity': data['quantity'],
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in netting endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@api_v2.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check key services
        container = current_app.config.get('DI_CONTAINER')
        
        health_status = {
            'status': 'healthy',
            'services': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check each service
        service_checks = [
            'inventory_analyzer',
            'production_pipeline',
            'enhanced_forecasting',
            'yarn_intelligence',
            'cache_manager'
        ]
        
        for service_name in service_checks:
            try:
                service = getattr(container, service_name)()
                health_status['services'][service_name] = 'healthy'
            except Exception as e:
                health_status['services'][service_name] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503