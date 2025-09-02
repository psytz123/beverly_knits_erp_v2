#!/usr/bin/env python3
"""
API Consolidation Middleware
Handles redirects from old endpoints to new consolidated ones
Provides feature flags for gradual migration
"""

from flask import redirect, request, jsonify, current_app
from functools import wraps
import logging
from typing import Dict, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Feature flags for API consolidation
CONSOLIDATION_FLAGS = {
    'inventory_consolidated': False,
    'forecasting_consolidated': False,
    'production_consolidated': False,
    'yarn_consolidated': False,
    'planning_consolidated': False,
    'cache_consolidated': False,
    'enable_redirects': True,
    'log_deprecated_usage': True,
    'block_deprecated_after': None  # Set to datetime to block deprecated endpoints
}

# Mapping of old endpoints to new consolidated ones
ENDPOINT_MAPPINGS = {
    # Inventory consolidation
    '/api/inventory-status': '/api/inventory/unified',
    '/api/yarn-inventory': '/api/inventory/unified',
    '/api/yarn-inventory-status': '/api/inventory/unified',
    '/api/production-inventory': '/api/inventory/unified',
    '/api/inventory-intelligence': '/api/inventory/unified',
    '/api/inventory-intelligence-enhanced': '/api/inventory/unified',
    
    # Forecasting consolidation
    '/api/forecast': '/api/forecast/unified',
    '/api/forecasting': '/api/forecast/unified',
    '/api/sales-forecast': '/api/forecast/unified',
    '/api/demand-forecast': '/api/forecast/unified',
    '/api/ml-forecast': '/api/forecast/unified',
    '/api/ml-forecast-report': '/api/forecast/unified',
    '/api/ml-forecast-detailed': '/api/forecast/unified',
    '/api/forecasted-orders': '/api/forecast/unified',
    
    # Production consolidation
    '/api/production-suggestions': '/api/production/unified',
    '/api/production-planning': '/api/production/unified',
    '/api/production-schedule': '/api/production/unified',
    '/api/production-orders': '/api/production/unified',
    '/api/production-status': '/api/production/unified',
    '/api/production-pipeline': '/api/production/unified',
    '/api/production-phases': '/api/production/unified',
    '/api/production-recommendations-ml': '/api/production/unified',
    '/api/ai-production-suggestions': '/api/production/unified',
    '/api/production-requirements': '/api/production/unified',
    '/api/production-capacity': '/api/production/unified',
    '/api/production-bottlenecks': '/api/production/unified',
    
    # Yarn consolidation
    '/api/yarn-intelligence': '/api/yarn/unified',
    '/api/yarn-shortages': '/api/yarn/unified',
    '/api/yarn-forecast-shortages': '/api/yarn/unified',
    '/api/yarn-aggregation': '/api/yarn/unified',
    '/api/yarn-alternatives': '/api/yarn/unified',
    '/api/yarn-substitution': '/api/yarn/unified',
    '/api/yarn-substitution-intelligent': '/api/yarn/unified',
    '/api/yarn-compatibility': '/api/yarn/unified',
    '/api/yarn-optimization': '/api/yarn/unified',
    
    # Planning consolidation
    '/api/six-phase-planning': '/api/planning/unified',
    '/api/planning-phases': '/api/planning/unified',
    '/api/planning-optimization': '/api/planning/unified',
    '/api/capacity-planning': '/api/planning/unified',
    '/api/resource-planning': '/api/planning/unified',
    '/api/supply-chain-planning': '/api/planning/unified',
    '/api/planning-intelligence': '/api/planning/unified',
    
    # Cache/Debug consolidation
    '/api/cache-stats': '/api/system/unified',
    '/api/debug-data': '/api/system/unified',
    '/api/reload-data': '/api/system/unified'
}

# Deprecated endpoints tracking
deprecated_usage = {}


def is_consolidation_enabled(category: str) -> bool:
    """Check if consolidation is enabled for a category"""
    flag_key = f"{category}_consolidated"
    return CONSOLIDATION_FLAGS.get(flag_key, False)


def log_deprecated_usage(old_endpoint: str, new_endpoint: str):
    """Log usage of deprecated endpoints"""
    if not CONSOLIDATION_FLAGS.get('log_deprecated_usage', True):
        return
    
    if old_endpoint not in deprecated_usage:
        deprecated_usage[old_endpoint] = {
            'count': 0,
            'first_seen': datetime.now(),
            'last_seen': None,
            'new_endpoint': new_endpoint
        }
    
    deprecated_usage[old_endpoint]['count'] += 1
    deprecated_usage[old_endpoint]['last_seen'] = datetime.now()
    
    logger.warning(f"Deprecated endpoint used: {old_endpoint} -> {new_endpoint} "
                  f"(Count: {deprecated_usage[old_endpoint]['count']})")


def redirect_deprecated(f: Callable) -> Callable:
    """Decorator to redirect deprecated endpoints to new ones"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not CONSOLIDATION_FLAGS.get('enable_redirects', True):
            return f(*args, **kwargs)
        
        # Check if current endpoint is deprecated
        current_path = request.path
        if current_path in ENDPOINT_MAPPINGS:
            new_endpoint = ENDPOINT_MAPPINGS[current_path]
            
            # Log the usage
            log_deprecated_usage(current_path, new_endpoint)
            
            # Check if we should block deprecated endpoints
            block_after = CONSOLIDATION_FLAGS.get('block_deprecated_after')
            if block_after and datetime.now() > block_after:
                return jsonify({
                    'error': 'This endpoint has been deprecated',
                    'message': f'Please use {new_endpoint} instead',
                    'deprecated_since': block_after.isoformat()
                }), 410  # HTTP 410 Gone
            
            # Redirect to new endpoint with query parameters
            query_string = request.query_string.decode('utf-8')
            redirect_url = new_endpoint
            if query_string:
                redirect_url += f"?{query_string}"
            
            # Add deprecation warning header
            response = redirect(redirect_url, code=308)  # Permanent redirect
            response.headers['X-Deprecated-Endpoint'] = current_path
            response.headers['X-New-Endpoint'] = new_endpoint
            response.headers['Deprecation'] = 'true'
            response.headers['Link'] = f'<{new_endpoint}>; rel="successor-version"'
            
            return response
        
        return f(*args, **kwargs)
    
    return decorated_function


def consolidation_wrapper(category: str):
    """Wrapper to handle consolidation based on feature flags"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if consolidation is enabled for this category
            if is_consolidation_enabled(category):
                # Use consolidated endpoint
                return f(*args, **kwargs)
            else:
                # Fall back to original implementation
                # This would call the original non-consolidated function
                original_func = getattr(current_app, f'original_{f.__name__}', None)
                if original_func:
                    return original_func(*args, **kwargs)
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def get_deprecation_report() -> Dict[str, Any]:
    """Get a report of deprecated endpoint usage"""
    report = {
        'total_deprecated_calls': sum(info['count'] for info in deprecated_usage.values()),
        'unique_deprecated_endpoints': len(deprecated_usage),
        'endpoints': []
    }
    
    for endpoint, info in deprecated_usage.items():
        report['endpoints'].append({
            'old_endpoint': endpoint,
            'new_endpoint': info['new_endpoint'],
            'usage_count': info['count'],
            'first_seen': info['first_seen'].isoformat() if info['first_seen'] else None,
            'last_seen': info['last_seen'].isoformat() if info['last_seen'] else None
        })
    
    # Sort by usage count
    report['endpoints'].sort(key=lambda x: x['usage_count'], reverse=True)
    
    return report


def update_feature_flag(flag_name: str, value: Any) -> bool:
    """Update a feature flag value"""
    if flag_name in CONSOLIDATION_FLAGS:
        old_value = CONSOLIDATION_FLAGS[flag_name]
        CONSOLIDATION_FLAGS[flag_name] = value
        logger.info(f"Feature flag '{flag_name}' updated: {old_value} -> {value}")
        return True
    return False


def get_feature_flags() -> Dict[str, Any]:
    """Get current feature flag values"""
    return CONSOLIDATION_FLAGS.copy()


class APIConsolidationMiddleware:
    """Middleware class for API consolidation"""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the middleware with Flask app"""
        self.app = app
        
        # Add before request handler
        app.before_request(self.check_deprecated_endpoint)
        
        # Add admin endpoints for consolidation management
        app.add_url_rule('/api/admin/deprecation-report', 
                        'deprecation_report', 
                        self.deprecation_report_endpoint,
                        methods=['GET'])
        app.add_url_rule('/api/admin/feature-flags',
                        'feature_flags',
                        self.feature_flags_endpoint,
                        methods=['GET', 'POST'])
    
    def check_deprecated_endpoint(self):
        """Check if the requested endpoint is deprecated"""
        if not CONSOLIDATION_FLAGS.get('enable_redirects', True):
            return None
        
        current_path = request.path
        if current_path in ENDPOINT_MAPPINGS:
            new_endpoint = ENDPOINT_MAPPINGS[current_path]
            log_deprecated_usage(current_path, new_endpoint)
            
            # Check if we should block
            block_after = CONSOLIDATION_FLAGS.get('block_deprecated_after')
            if block_after and datetime.now() > block_after:
                return jsonify({
                    'error': 'This endpoint has been deprecated',
                    'message': f'Please use {new_endpoint} instead',
                    'deprecated_since': block_after.isoformat()
                }), 410
            
            # Add warning headers but don't redirect automatically
            # This allows gradual migration
            @wraps(request)
            def add_deprecation_headers(response):
                response.headers['X-Deprecated-Endpoint'] = current_path
                response.headers['X-New-Endpoint'] = new_endpoint
                response.headers['Deprecation'] = 'true'
                response.headers['Link'] = f'<{new_endpoint}>; rel="successor-version"'
                response.headers['Warning'] = f'299 - "This endpoint is deprecated. Use {new_endpoint}"'
                return response
            
            request.after_request_funcs = getattr(request, 'after_request_funcs', [])
            request.after_request_funcs.append(add_deprecation_headers)
        
        return None
    
    def deprecation_report_endpoint(self):
        """Endpoint to get deprecation report"""
        return jsonify(get_deprecation_report())
    
    def feature_flags_endpoint(self):
        """Endpoint to manage feature flags"""
        if request.method == 'GET':
            return jsonify(get_feature_flags())
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            updated = {}
            for flag_name, value in data.items():
                if update_feature_flag(flag_name, value):
                    updated[flag_name] = value
            
            return jsonify({
                'updated': updated,
                'current_flags': get_feature_flags()
            })


# Export key functions and classes
__all__ = [
    'APIConsolidationMiddleware',
    'redirect_deprecated',
    'consolidation_wrapper',
    'get_deprecation_report',
    'update_feature_flag',
    'get_feature_flags',
    'is_consolidation_enabled',
    'ENDPOINT_MAPPINGS',
    'CONSOLIDATION_FLAGS'
]