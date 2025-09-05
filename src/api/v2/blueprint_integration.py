#!/usr/bin/env python3
"""
Blueprint Integration Module - Phase 4
Handles registration and initialization of API v2 blueprints
"""

from flask import Flask, redirect, request, jsonify
from typing import Dict, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Import the v2 blueprint
from src.api.v2.consolidated_routes import api_v2

# Deprecated endpoint mappings for backward compatibility
DEPRECATED_ENDPOINT_MAPPINGS = {
    # Inventory endpoints
    '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
    '/api/yarn-data': '/api/v2/inventory?view=yarn',
    '/api/inventory-intelligence-enhanced': '/api/v2/inventory?analysis=intelligence',
    '/api/real-time-inventory-dashboard': '/api/v2/inventory?realtime=true',
    '/api/emergency-shortage-dashboard': '/api/v2/inventory?view=shortage&shortage_only=true',
    '/api/inventory-netting': '/api/v2/inventory?view=planning',
    '/api/yarn-coverage': '/api/v2/inventory?analysis=shortage',
    
    # Production endpoints  
    '/api/production-planning': '/api/v2/production?view=planning',
    '/api/production-status': '/api/v2/production?view=status',
    '/api/production-pipeline': '/api/v2/production?view=pipeline',
    '/api/production-recommendations-ml': '/api/v2/production?view=recommendations',
    '/api/machine-assignment-suggestions': '/api/v2/production?view=machines',
    '/api/production-flow': '/api/v2/production?view=pipeline',
    '/api/knit-orders': '/api/v2/production?view=status',
    '/api/production-suggestions': '/api/v2/production?view=recommendations',
    
    # Forecasting endpoints
    '/api/ml-forecasting': '/api/v2/forecast',
    '/api/ml-forecast-detailed': '/api/v2/forecast?detail=full',
    '/api/sales-forecasting': '/api/v2/forecast?model=prophet',
    '/api/demand-forecast': '/api/v2/forecast',
    '/api/forecast-accuracy': '/api/v2/forecast?detail=accuracy',
    '/api/ml-forecast': '/api/v2/forecast',
    '/api/sales-forecast': '/api/v2/forecast?model=prophet',
    
    # Analytics endpoints
    '/api/comprehensive-kpis': '/api/v2/analytics?category=kpi',
    '/api/business-metrics': '/api/v2/analytics?category=business',
    '/api/performance-metrics': '/api/v2/analytics?category=performance',
    '/api/analytics-dashboard': '/api/v2/analytics',
    '/api/real-time-metrics': '/api/v2/analytics?realtime=true',
    '/api/kpi-dashboard': '/api/v2/analytics?category=kpi',
    
    # Yarn management endpoints
    '/api/yarn-intelligence': '/api/v2/yarn?action=intelligence',
    '/api/yarn-substitution-intelligent': '/api/v2/yarn?action=substitution',
    '/api/yarn-interchangeability': '/api/v2/yarn?action=substitution',
    '/api/yarn-requirements': '/api/v2/yarn?action=requirements',
    '/api/yarn-shortage-analysis': '/api/v2/yarn?action=intelligence&include_substitutes=true',
    
    # Factory floor endpoints
    '/api/factory-floor-ai-dashboard': '/api/v2/production?view=machines',
    '/api/machine-planning': '/api/v2/production?view=machines',
    
    # Risk and optimization
    '/api/po-risk-analysis': '/api/v2/production?view=planning&include_forecast=true',
    '/api/supply-chain-optimization': '/api/v2/analytics?category=business',
    
    # Capacity planning
    '/api/capacity-planning': '/api/v2/production?view=planning',
    '/api/capacity-optimization': '/api/v2/production?view=planning&include_forecast=true',
}

def create_redirect_handler(new_endpoint: str):
    """Create a redirect handler for deprecated endpoints with parameter mapping"""
    def handler():
        from src.api.v2.parameter_mapper import ParameterMapper
        
        # Get original endpoint name for parameter mapping
        endpoint_name = request.path.replace('/api/', '').strip('/')
        
        # Get query parameters from original request
        query_params = request.args.to_dict()
        
        # Map old parameters to new format
        mapped_params = ParameterMapper.map_parameters(endpoint_name, query_params)
        
        # Parse new endpoint to extract base path and default params
        if '?' in new_endpoint:
            base_path, default_params_str = new_endpoint.split('?', 1)
            # Parse default parameters
            default_params = {}
            for param in default_params_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    default_params[key] = value
            
            # Merge parameters (mapped params override defaults)
            final_params = {**default_params, **mapped_params}
        else:
            base_path = new_endpoint
            final_params = mapped_params
        
        # Build final URL with parameters
        if final_params:
            param_str = '&'.join([f"{k}={v}" for k, v in final_params.items()])
            redirect_url = f"{base_path}?{param_str}"
        else:
            redirect_url = base_path
        
        # Log deprecated usage with parameter mapping
        logger.warning(f"Deprecated endpoint accessed: {request.path} -> {redirect_url}")
        if query_params != mapped_params:
            logger.info(f"Parameters mapped: {query_params} -> {mapped_params}")
        
        # Perform redirect
        return redirect(redirect_url, code=308)  # 308 = Permanent Redirect preserving method
    
    return handler

def register_blueprints(app: Flask, enable_deprecation_redirects: bool = True):
    """
    Register API v2 blueprints with the Flask application
    
    Args:
        app: Flask application instance
        enable_deprecation_redirects: Whether to enable backward compatibility redirects
    """
    try:
        # Register the v2 API blueprint
        app.register_blueprint(api_v2)
        logger.info("API v2 blueprint registered successfully")
        
        # Register deprecation redirects if enabled
        if enable_deprecation_redirects:
            register_deprecation_redirects(app)
            logger.info(f"Registered {len(DEPRECATED_ENDPOINT_MAPPINGS)} deprecation redirects")
        
        # Add consolidation metrics endpoint
        @app.route('/api/consolidation-metrics', methods=['GET'])
        def consolidation_metrics():
            """Return API consolidation metrics"""
            return jsonify({
                'consolidated_endpoints': 25,
                'deprecated_endpoints': len(DEPRECATED_ENDPOINT_MAPPINGS),
                'reduction_percentage': 73.7,  # (95-25)/95 * 100
                'v2_adoption_rate': get_v2_adoption_rate(),
                'deprecation_redirects_enabled': enable_deprecation_redirects
            })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register blueprints: {str(e)}")
        return False

def register_deprecation_redirects(app: Flask):
    """Register redirect handlers for all deprecated endpoints"""
    for old_endpoint, new_endpoint in DEPRECATED_ENDPOINT_MAPPINGS.items():
        # Create unique function name for each handler
        endpoint_name = f"deprecated_{old_endpoint.replace('/', '_').replace('-', '_')}"
        
        # Register the redirect handler
        app.add_url_rule(
            old_endpoint,
            endpoint=endpoint_name,
            view_func=create_redirect_handler(new_endpoint),
            methods=['GET', 'POST', 'PUT', 'DELETE']
        )

def get_v2_adoption_rate() -> float:
    """Calculate the adoption rate of v2 endpoints"""
    # This would normally track actual usage metrics
    # For now, return a placeholder value
    return 0.45  # 45% adoption rate

def init_api_v2(app: Flask) -> bool:
    """
    Initialize API v2 with all features
    
    Args:
        app: Flask application instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if feature flags are available
        try:
            from src.config.feature_flags import is_consolidation_enabled
            consolidation_enabled = is_consolidation_enabled()
        except ImportError:
            consolidation_enabled = True  # Default to enabled if no feature flags
        
        if not consolidation_enabled:
            logger.info("API consolidation disabled by feature flag")
            return False
        
        # Initialize dependency injection
        try:
            from src.api.v2.dependency_injection import init_dependency_injection
            init_dependency_injection(app)
            logger.info("Dependency injection initialized")
        except ImportError as e:
            logger.warning(f"Dependency injection not available: {e}")
        
        # Register blueprints with deprecation handling
        success = register_blueprints(app, enable_deprecation_redirects=True)
        
        if success:
            logger.info("API v2 initialization complete")
            
            # Add middleware for request tracking
            @app.before_request
            def track_api_requests():
                """Track API request patterns for analytics"""
                if request.path.startswith('/api/v2/'):
                    # Track v2 usage
                    logger.debug(f"V2 API request: {request.path}")
                elif request.path.startswith('/api/') and request.path in DEPRECATED_ENDPOINT_MAPPINGS:
                    # Track deprecated endpoint usage
                    logger.warning(f"Deprecated endpoint used: {request.path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to initialize API v2: {str(e)}")
        return False

# Export main functions
__all__ = [
    'register_blueprints',
    'init_api_v2',
    'DEPRECATED_ENDPOINT_MAPPINGS'
]