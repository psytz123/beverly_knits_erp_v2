#!/usr/bin/env python3
"""
Consolidated API Endpoints
Unified endpoints that combine functionality from multiple redundant APIs
"""

from flask import jsonify, request, current_app
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class ConsolidatedInventoryAPI:
    """Consolidated inventory endpoint combining multiple inventory APIs"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified inventory endpoint that handles all inventory-related queries
        Combines: inventory-status, yarn-inventory, production-inventory, 
                 inventory-intelligence, inventory-intelligence-enhanced
        """
        try:
            # Get query parameters to determine what data to return
            query_type = request.args.get('type', 'all')
            include_intelligence = request.args.get('intelligence', 'true').lower() == 'true'
            include_yarn = request.args.get('yarn', 'true').lower() == 'true'
            include_production = request.args.get('production', 'true').lower() == 'true'
            style_filter = request.args.get('style')
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            # Get yarn inventory if requested
            if include_yarn or query_type in ['yarn', 'all']:
                try:
                    yarn_data = erp.get_yarn_inventory_status()
                    response['data']['yarn'] = yarn_data
                except Exception as e:
                    logger.error(f"Error getting yarn inventory: {e}")
                    response['data']['yarn'] = {'error': str(e)}
            
            # Get production inventory if requested
            if include_production or query_type in ['production', 'all']:
                try:
                    prod_data = erp.get_production_inventory()
                    response['data']['production'] = prod_data
                except Exception as e:
                    logger.error(f"Error getting production inventory: {e}")
                    response['data']['production'] = {'error': str(e)}
            
            # Get intelligence metrics if requested
            if include_intelligence or query_type in ['intelligence', 'all']:
                try:
                    intel_data = erp.get_inventory_intelligence_enhanced()
                    response['data']['intelligence'] = intel_data
                except Exception as e:
                    logger.error(f"Error getting inventory intelligence: {e}")
                    response['data']['intelligence'] = {'error': str(e)}
            
            # Apply style filter if provided
            if style_filter and 'production' in response['data']:
                filtered = [item for item in response['data']['production'].get('items', [])
                           if style_filter.lower() in str(item.get('style', '')).lower()]
                response['data']['production']['items'] = filtered
                response['data']['production']['filtered_by'] = style_filter
            
            # Add summary statistics
            response['summary'] = {
                'total_yarn_items': len(response['data'].get('yarn', {}).get('inventory', [])),
                'total_production_items': len(response['data'].get('production', {}).get('items', [])),
                'has_intelligence': 'intelligence' in response['data'],
                'query_type': query_type,
                'filters_applied': bool(style_filter)
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified inventory endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to get inventory data',
                'message': str(e)
            }), 500


class ConsolidatedForecastAPI:
    """Consolidated forecasting endpoint combining multiple forecast APIs"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified forecast endpoint that handles all forecasting queries
        Combines: forecast, forecasting, sales-forecast, demand-forecast,
                 ml-forecast, ml-forecast-report, ml-forecast-detailed, forecasted-orders
        """
        try:
            # Get query parameters
            forecast_type = request.args.get('type', 'all')
            horizon_days = int(request.args.get('horizon', 90))
            include_ml = request.args.get('ml', 'true').lower() == 'true'
            include_details = request.args.get('details', 'false').lower() == 'true'
            style_filter = request.args.get('style')
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'horizon_days': horizon_days,
                'data': {}
            }
            
            # Get sales forecast
            if forecast_type in ['sales', 'all']:
                try:
                    sales_forecast = erp.get_sales_forecast(horizon_days)
                    response['data']['sales'] = sales_forecast
                except Exception as e:
                    logger.error(f"Error getting sales forecast: {e}")
                    response['data']['sales'] = {'error': str(e)}
            
            # Get demand forecast
            if forecast_type in ['demand', 'all']:
                try:
                    demand_forecast = erp.get_demand_forecast()
                    response['data']['demand'] = demand_forecast
                except Exception as e:
                    logger.error(f"Error getting demand forecast: {e}")
                    response['data']['demand'] = {'error': str(e)}
            
            # Get ML forecast if requested
            if include_ml or forecast_type in ['ml', 'all']:
                try:
                    if include_details:
                        ml_forecast = erp.get_ml_forecast_detailed()
                    else:
                        ml_forecast = erp.get_ml_forecast_report()
                    response['data']['ml'] = ml_forecast
                except Exception as e:
                    logger.error(f"Error getting ML forecast: {e}")
                    response['data']['ml'] = {'error': str(e)}
            
            # Get forecasted orders
            if forecast_type in ['orders', 'all']:
                try:
                    orders = erp.get_forecasted_orders()
                    response['data']['orders'] = orders
                except Exception as e:
                    logger.error(f"Error getting forecasted orders: {e}")
                    response['data']['orders'] = {'error': str(e)}
            
            # Apply style filter if provided
            if style_filter:
                for key in ['sales', 'demand', 'orders']:
                    if key in response['data'] and 'items' in response['data'][key]:
                        filtered = [item for item in response['data'][key]['items']
                                   if style_filter.lower() in str(item.get('style', '')).lower()]
                        response['data'][key]['items'] = filtered
                        response['data'][key]['filtered_by'] = style_filter
            
            # Add summary
            response['summary'] = {
                'forecast_type': forecast_type,
                'includes_ml': include_ml,
                'includes_details': include_details,
                'filters_applied': bool(style_filter),
                'data_types_returned': list(response['data'].keys())
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified forecast endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to get forecast data',
                'message': str(e)
            }), 500


class ConsolidatedProductionAPI:
    """Consolidated production endpoint combining multiple production APIs"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified production endpoint that handles all production queries
        Combines: production-suggestions, production-planning, production-schedule,
                 production-orders, production-status, production-pipeline, etc.
        """
        try:
            # Get query parameters
            query_type = request.args.get('type', 'all')
            include_ml = request.args.get('ml', 'true').lower() == 'true'
            include_pipeline = request.args.get('pipeline', 'true').lower() == 'true'
            include_suggestions = request.args.get('suggestions', 'true').lower() == 'true'
            order_filter = request.args.get('order')
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            # Get production planning
            if query_type in ['planning', 'all']:
                try:
                    planning = erp.get_production_planning()
                    response['data']['planning'] = planning
                except Exception as e:
                    logger.error(f"Error getting production planning: {e}")
                    response['data']['planning'] = {'error': str(e)}
            
            # Get production pipeline
            if include_pipeline or query_type in ['pipeline', 'all']:
                try:
                    pipeline = erp.get_production_pipeline()
                    response['data']['pipeline'] = pipeline
                except Exception as e:
                    logger.error(f"Error getting production pipeline: {e}")
                    response['data']['pipeline'] = {'error': str(e)}
            
            # Get AI suggestions
            if include_suggestions or query_type in ['suggestions', 'all']:
                try:
                    suggestions = erp.get_ai_production_suggestions()
                    response['data']['suggestions'] = suggestions
                except Exception as e:
                    logger.error(f"Error getting AI suggestions: {e}")
                    response['data']['suggestions'] = {'error': str(e)}
            
            # Get ML recommendations if requested
            if include_ml or query_type in ['ml', 'all']:
                try:
                    ml_recommendations = erp.get_production_recommendations_ml()
                    response['data']['ml_recommendations'] = ml_recommendations
                except Exception as e:
                    logger.error(f"Error getting ML recommendations: {e}")
                    response['data']['ml_recommendations'] = {'error': str(e)}
            
            # Get production status
            if query_type in ['status', 'all']:
                try:
                    status = erp.get_production_status()
                    response['data']['status'] = status
                except Exception as e:
                    logger.error(f"Error getting production status: {e}")
                    response['data']['status'] = {'error': str(e)}
            
            # Apply order filter if provided
            if order_filter:
                for key in response['data']:
                    if isinstance(response['data'][key], dict) and 'orders' in response['data'][key]:
                        filtered = [order for order in response['data'][key]['orders']
                                   if order_filter.lower() in str(order.get('order_id', '')).lower()]
                        response['data'][key]['orders'] = filtered
                        response['data'][key]['filtered_by'] = order_filter
            
            # Add summary
            response['summary'] = {
                'query_type': query_type,
                'includes_ml': include_ml,
                'includes_pipeline': include_pipeline,
                'includes_suggestions': include_suggestions,
                'filters_applied': bool(order_filter),
                'data_types_returned': list(response['data'].keys())
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified production endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to get production data',
                'message': str(e)
            }), 500


class ConsolidatedYarnAPI:
    """Consolidated yarn endpoint combining multiple yarn APIs"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified yarn endpoint that handles all yarn-related queries
        Combines: yarn-intelligence, yarn-shortages, yarn-forecast-shortages,
                 yarn-aggregation, yarn-alternatives, yarn-substitution, etc.
        """
        try:
            # Get query parameters
            query_type = request.args.get('type', 'all')
            include_intelligence = request.args.get('intelligence', 'true').lower() == 'true'
            include_substitutions = request.args.get('substitutions', 'true').lower() == 'true'
            include_shortages = request.args.get('shortages', 'true').lower() == 'true'
            yarn_filter = request.args.get('yarn')
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            # Get yarn intelligence
            if include_intelligence or query_type in ['intelligence', 'all']:
                try:
                    intelligence = erp.get_yarn_intelligence()
                    response['data']['intelligence'] = intelligence
                except Exception as e:
                    logger.error(f"Error getting yarn intelligence: {e}")
                    response['data']['intelligence'] = {'error': str(e)}
            
            # Get yarn shortages
            if include_shortages or query_type in ['shortages', 'all']:
                try:
                    shortages = erp.get_yarn_shortages()
                    forecast_shortages = erp.get_yarn_forecast_shortages()
                    response['data']['shortages'] = {
                        'current': shortages,
                        'forecasted': forecast_shortages
                    }
                except Exception as e:
                    logger.error(f"Error getting yarn shortages: {e}")
                    response['data']['shortages'] = {'error': str(e)}
            
            # Get yarn substitutions
            if include_substitutions or query_type in ['substitutions', 'all']:
                try:
                    alternatives = erp.get_yarn_alternatives()
                    intelligent_subs = erp.get_yarn_substitution_intelligent()
                    response['data']['substitutions'] = {
                        'alternatives': alternatives,
                        'intelligent': intelligent_subs
                    }
                except Exception as e:
                    logger.error(f"Error getting yarn substitutions: {e}")
                    response['data']['substitutions'] = {'error': str(e)}
            
            # Get yarn aggregation
            if query_type in ['aggregation', 'all']:
                try:
                    aggregation = erp.get_yarn_aggregation()
                    response['data']['aggregation'] = aggregation
                except Exception as e:
                    logger.error(f"Error getting yarn aggregation: {e}")
                    response['data']['aggregation'] = {'error': str(e)}
            
            # Apply yarn filter if provided
            if yarn_filter:
                for key in response['data']:
                    if isinstance(response['data'][key], dict):
                        # Filter nested yarn data
                        if 'yarns' in response['data'][key]:
                            filtered = [yarn for yarn in response['data'][key]['yarns']
                                       if yarn_filter.lower() in str(yarn.get('id', '')).lower()]
                            response['data'][key]['yarns'] = filtered
                            response['data'][key]['filtered_by'] = yarn_filter
            
            # Add summary
            response['summary'] = {
                'query_type': query_type,
                'includes_intelligence': include_intelligence,
                'includes_substitutions': include_substitutions,
                'includes_shortages': include_shortages,
                'filters_applied': bool(yarn_filter),
                'data_types_returned': list(response['data'].keys())
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified yarn endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to get yarn data',
                'message': str(e)
            }), 500


class ConsolidatedPlanningAPI:
    """Consolidated planning endpoint combining multiple planning APIs"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified planning endpoint that handles all planning queries
        Combines: six-phase-planning, planning-phases, planning-optimization,
                 capacity-planning, resource-planning, supply-chain-planning
        """
        try:
            # Get query parameters
            planning_type = request.args.get('type', 'all')
            include_optimization = request.args.get('optimization', 'true').lower() == 'true'
            include_capacity = request.args.get('capacity', 'true').lower() == 'true'
            phase_filter = request.args.get('phase')
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            # Get six-phase planning
            if planning_type in ['six-phase', 'all']:
                try:
                    six_phase = erp.get_six_phase_planning()
                    response['data']['six_phase'] = six_phase
                except Exception as e:
                    logger.error(f"Error getting six-phase planning: {e}")
                    response['data']['six_phase'] = {'error': str(e)}
            
            # Get capacity planning
            if include_capacity or planning_type in ['capacity', 'all']:
                try:
                    capacity = erp.get_capacity_planning()
                    response['data']['capacity'] = capacity
                except Exception as e:
                    logger.error(f"Error getting capacity planning: {e}")
                    response['data']['capacity'] = {'error': str(e)}
            
            # Get planning optimization
            if include_optimization or planning_type in ['optimization', 'all']:
                try:
                    optimization = erp.get_planning_optimization()
                    response['data']['optimization'] = optimization
                except Exception as e:
                    logger.error(f"Error getting planning optimization: {e}")
                    response['data']['optimization'] = {'error': str(e)}
            
            # Apply phase filter if provided
            if phase_filter and 'six_phase' in response['data']:
                if 'phases' in response['data']['six_phase']:
                    filtered_phases = {k: v for k, v in response['data']['six_phase']['phases'].items()
                                      if phase_filter.lower() in k.lower()}
                    response['data']['six_phase']['phases'] = filtered_phases
                    response['data']['six_phase']['filtered_by'] = phase_filter
            
            # Add summary
            response['summary'] = {
                'planning_type': planning_type,
                'includes_optimization': include_optimization,
                'includes_capacity': include_capacity,
                'filters_applied': bool(phase_filter),
                'data_types_returned': list(response['data'].keys())
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified planning endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to get planning data',
                'message': str(e)
            }), 500


class ConsolidatedSystemAPI:
    """Consolidated system endpoint for cache and debug operations"""
    
    @staticmethod
    def unified_endpoint():
        """
        Unified system endpoint for cache, debug, and data operations
        Combines: cache-stats, debug-data, reload-data
        """
        try:
            # Get query parameters
            operation = request.args.get('operation', 'status')
            force_reload = request.args.get('force_reload', 'false').lower() == 'true'
            
            # Get the ERP instance
            erp = current_app.config.get('erp_instance')
            if not erp:
                return jsonify({'error': 'ERP system not initialized'}), 500
            
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'data': {}
            }
            
            # Handle different operations
            if operation == 'reload' or force_reload:
                try:
                    result = erp.reload_data()
                    response['data']['reload'] = result
                    response['message'] = 'Data reloaded successfully'
                except Exception as e:
                    logger.error(f"Error reloading data: {e}")
                    response['data']['reload'] = {'error': str(e)}
                    response['message'] = 'Failed to reload data'
            
            if operation in ['cache', 'status', 'all']:
                try:
                    cache_stats = erp.get_cache_stats()
                    response['data']['cache'] = cache_stats
                except Exception as e:
                    logger.error(f"Error getting cache stats: {e}")
                    response['data']['cache'] = {'error': str(e)}
            
            if operation in ['debug', 'all']:
                try:
                    debug_data = erp.get_debug_data()
                    response['data']['debug'] = debug_data
                except Exception as e:
                    logger.error(f"Error getting debug data: {e}")
                    response['data']['debug'] = {'error': str(e)}
            
            # Add system health info
            response['system'] = {
                'healthy': all('error' not in v for v in response['data'].values() if isinstance(v, dict)),
                'operations_completed': list(response['data'].keys()),
                'force_reload': force_reload
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in unified system endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to execute system operation',
                'message': str(e)
            }), 500


def register_consolidated_endpoints(app, erp_instance):
    """Register all consolidated endpoints with the Flask app"""
    
    # Store ERP instance in app config
    app.config['erp_instance'] = erp_instance
    
    # Register inventory endpoint
    app.add_url_rule('/api/inventory/unified',
                     'inventory_unified',
                     ConsolidatedInventoryAPI.unified_endpoint,
                     methods=['GET'])
    
    # Register forecast endpoint
    app.add_url_rule('/api/forecast/unified',
                     'forecast_unified',
                     ConsolidatedForecastAPI.unified_endpoint,
                     methods=['GET'])
    
    # Register production endpoint
    app.add_url_rule('/api/production/unified',
                     'production_unified',
                     ConsolidatedProductionAPI.unified_endpoint,
                     methods=['GET'])
    
    # Register yarn endpoint
    app.add_url_rule('/api/yarn/unified',
                     'yarn_unified',
                     ConsolidatedYarnAPI.unified_endpoint,
                     methods=['GET'])
    
    # Register planning endpoint
    app.add_url_rule('/api/planning/unified',
                     'planning_unified',
                     ConsolidatedPlanningAPI.unified_endpoint,
                     methods=['GET'])
    
    # Register system endpoint
    app.add_url_rule('/api/system/unified',
                     'system_unified',
                     ConsolidatedSystemAPI.unified_endpoint,
                     methods=['GET', 'POST'])
    
    logger.info("Consolidated endpoints registered successfully")
    return True


# Export classes and functions
__all__ = [
    'ConsolidatedInventoryAPI',
    'ConsolidatedForecastAPI',
    'ConsolidatedProductionAPI',
    'ConsolidatedYarnAPI',
    'ConsolidatedPlanningAPI',
    'ConsolidatedSystemAPI',
    'register_consolidated_endpoints'
]