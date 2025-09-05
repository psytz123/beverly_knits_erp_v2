"""
API Backward Compatibility Layer
Maintains backward compatibility for deprecated endpoints
"""

from flask import redirect, request, jsonify
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class CompatibilityLayer:
    """Handles backward compatibility for deprecated API endpoints"""
    
    # Mapping of old endpoints to new v2 endpoints
    LEGACY_MAPPINGS = {
        # Inventory endpoints
        '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
        '/api/inventory-status': '/api/v2/inventory?view=summary',
        '/api/inventory-details': '/api/v2/inventory?view=detailed',
        '/api/yarn-shortages': '/api/v2/inventory/shortages',
        '/api/planning-balance': '/api/v2/inventory/planning-balance',
        '/api/inventory-intelligence': '/api/v2/inventory',
        '/api/inventory-intelligence-enhanced': '/api/v2/inventory?view=detailed&analysis=all',
        
        # Production endpoints
        '/api/production-status': '/api/v2/production/status',
        '/api/production-schedule': '/api/v2/production/planning?view=schedule',
        '/api/production-capacity': '/api/v2/production/capacity',
        '/api/production-orders': '/api/v2/production/orders',
        '/api/machine-utilization': '/api/v2/production/status',
        '/api/production-planning': '/api/v2/production/planning',
        '/api/production-recommendations': '/api/v2/production/recommendations',
        '/api/machine-assignment-suggestions': '/api/v2/production/assign-machine',
        
        # Yarn endpoints
        '/api/yarn-requirements': '/api/v2/yarn/requirements',
        '/api/yarn-substitution': '/api/v2/yarn/substitution',
        '/api/yarn-availability': '/api/v2/yarn/availability-check',
        '/api/yarn-intelligence': '/api/v2/yarn/intelligence',
        '/api/yarn-substitution-intelligent': '/api/v2/yarn/substitution',
        '/api/yarn-interchangeability': '/api/v2/yarn/interchangeability',
        
        # Forecasting endpoints
        '/api/forecast-demand': '/api/v2/forecasting/demand',
        '/api/ml-forecast': '/api/v2/forecasting/ml-detailed',
        '/api/ml-forecast-detailed': '/api/v2/forecasting/ml-detailed',
        '/api/forecast-accuracy': '/api/v2/forecasting/accuracy',
        '/api/seasonal-analysis': '/api/v2/forecasting/seasonal-analysis',
        
        # Supply chain endpoints
        '/api/supply-chain-analysis': '/api/v2/supply-chain/analysis',
        '/api/mrp-requirements': '/api/v2/mrp/requirements',
        '/api/po-risk-analysis': '/api/v2/po-risk-analysis',
        '/api/supply-chain-kpis': '/api/v2/supply-chain/kpis',
        '/api/comprehensive-kpis': '/api/v2/supply-chain/kpis',
        
        # Other endpoints
        '/api/inventory-netting': '/api/v2/inventory/netting',
        '/api/production-pipeline': '/api/v2/production/planning?view=pipeline',
        '/api/production-suggestions': '/api/v2/production/recommendations',
        '/api/production-recommendations-ml': '/api/v2/production/recommendations',
        '/api/knit-orders': '/api/v2/production/orders',
        '/api/factory-floor-ai-dashboard': '/api/v2/production/status'
    }
    
    # Deprecated endpoints that should show warning
    DEPRECATED_ENDPOINTS = set(LEGACY_MAPPINGS.keys())
    
    # Metrics for tracking deprecated usage
    deprecation_metrics = {}
    
    @classmethod
    def create_legacy_redirects(cls, app):
        """
        Create redirect routes for all legacy endpoints
        
        Args:
            app: Flask application instance
        """
        for old_path, new_path in cls.LEGACY_MAPPINGS.items():
            cls._create_redirect_route(app, old_path, new_path)
        
        logger.info(f"Created {len(cls.LEGACY_MAPPINGS)} legacy redirect routes")
    
    @classmethod
    def _create_redirect_route(cls, app, old_path: str, new_path: str):
        """
        Create a single redirect route
        
        Args:
            app: Flask application instance
            old_path: Old endpoint path
            new_path: New v2 endpoint path
        """
        def redirect_handler():
            """Handler for deprecated endpoint"""
            # Track usage
            cls._track_deprecated_usage(old_path)
            
            # Log warning
            logger.warning(f"Deprecated endpoint accessed: {old_path} -> {new_path}")
            
            # Build new URL with query parameters
            full_new_path = new_path
            if request.args:
                # Preserve query parameters
                query_string = request.query_string.decode('utf-8')
                separator = '&' if '?' in new_path else '?'
                full_new_path = f"{new_path}{separator}{query_string}"
            
            # Return deprecation warning with redirect
            if request.method == 'GET':
                response = redirect(full_new_path, code=307)
                response.headers['X-Deprecated'] = 'true'
                response.headers['X-New-Endpoint'] = new_path
                return response
            else:
                # For non-GET requests, proxy the request
                return cls._proxy_request(new_path)
        
        # Register the route
        endpoint_name = f"legacy_{old_path.replace('/', '_')}"
        app.add_url_rule(old_path, endpoint_name, redirect_handler, methods=['GET', 'POST', 'PUT', 'DELETE'])
    
    @classmethod
    def _track_deprecated_usage(cls, endpoint: str):
        """Track usage of deprecated endpoints"""
        if endpoint not in cls.deprecation_metrics:
            cls.deprecation_metrics[endpoint] = 0
        cls.deprecation_metrics[endpoint] += 1
    
    @classmethod
    def _proxy_request(cls, new_path: str):
        """
        Proxy non-GET requests to new endpoint
        
        Args:
            new_path: New endpoint path
            
        Returns:
            Response from new endpoint
        """
        try:
            # Import here to avoid circular dependency
            from flask import current_app
            
            # Build full URL
            with current_app.test_client() as client:
                # Get request data
                data = request.get_json(silent=True)
                
                # Make request to new endpoint
                if request.method == 'POST':
                    response = client.post(new_path, json=data)
                elif request.method == 'PUT':
                    response = client.put(new_path, json=data)
                elif request.method == 'DELETE':
                    response = client.delete(new_path)
                else:
                    response = client.get(new_path)
                
                # Add deprecation headers
                response.headers['X-Deprecated'] = 'true'
                response.headers['X-New-Endpoint'] = new_path
                
                return response
                
        except Exception as e:
            logger.error(f"Error proxying request to {new_path}: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to proxy request to new endpoint',
                'new_endpoint': new_path
            }), 500
    
    @classmethod
    def get_deprecation_metrics(cls) -> Dict[str, int]:
        """
        Get metrics on deprecated endpoint usage
        
        Returns:
            Dictionary of endpoint usage counts
        """
        return cls.deprecation_metrics.copy()
    
    @classmethod
    def intercept_deprecated_endpoints(cls):
        """
        Middleware to intercept and handle deprecated endpoints
        
        Returns:
            Middleware function
        """
        def middleware(environ, start_response):
            path = environ.get('PATH_INFO', '')
            
            if path in cls.DEPRECATED_ENDPOINTS:
                # Add deprecation warning header
                def custom_start_response(status, headers):
                    headers.append(('X-Deprecation-Warning', 
                                  f'This endpoint is deprecated. Use {cls.LEGACY_MAPPINGS.get(path, "v2 API")}'))
                    return start_response(status, headers)
                
                return custom_start_response
            
            return start_response
        
        return middleware
    
    @classmethod
    def generate_migration_guide(cls) -> Dict[str, Dict[str, str]]:
        """
        Generate migration guide for API consumers
        
        Returns:
            Migration guide with old and new endpoints
        """
        guide = {
            'endpoints': {},
            'summary': {
                'total_deprecated': len(cls.LEGACY_MAPPINGS),
                'categories': {
                    'inventory': 0,
                    'production': 0,
                    'yarn': 0,
                    'forecasting': 0,
                    'supply_chain': 0
                }
            }
        }
        
        for old, new in cls.LEGACY_MAPPINGS.items():
            # Categorize endpoint
            category = 'other'
            if 'inventory' in old or 'yarn-inventory' in old:
                category = 'inventory'
            elif 'production' in old or 'machine' in old:
                category = 'production'
            elif 'yarn' in old and 'inventory' not in old:
                category = 'yarn'
            elif 'forecast' in old or 'ml' in old:
                category = 'forecasting'
            elif 'supply-chain' in old or 'mrp' in old:
                category = 'supply_chain'
            
            guide['endpoints'][old] = {
                'new_endpoint': new,
                'category': category,
                'status': 'deprecated',
                'migration_notes': cls._get_migration_notes(old, new)
            }
            
            guide['summary']['categories'][category] = guide['summary']['categories'].get(category, 0) + 1
        
        return guide
    
    @classmethod
    def _get_migration_notes(cls, old_endpoint: str, new_endpoint: str) -> str:
        """
        Get migration notes for specific endpoint
        
        Args:
            old_endpoint: Old endpoint path
            new_endpoint: New endpoint path
            
        Returns:
            Migration notes string
        """
        notes = []
        
        # Check for parameter changes
        if '?' in new_endpoint:
            notes.append("Default parameters have been added")
        
        # Check for path structure changes
        if '/v2/' in new_endpoint:
            notes.append("Endpoint moved to v2 API structure")
        
        # Specific endpoint notes
        if 'intelligence' in old_endpoint:
            notes.append("Enhanced features available in new endpoint")
        
        if 'ml' in old_endpoint:
            notes.append("Additional ML models available")
        
        return ". ".join(notes) if notes else "Direct migration, no changes required"


def register_compatibility_layer(app):
    """
    Register compatibility layer with Flask app
    
    Args:
        app: Flask application instance
    """
    # Create legacy redirects
    CompatibilityLayer.create_legacy_redirects(app)
    
    # Add middleware for deprecation warnings
    app.wsgi_app = CompatibilityLayer.intercept_deprecated_endpoints()
    
    # Add metrics endpoint
    @app.route('/api/v2/deprecation-metrics')
    def get_deprecation_metrics():
        """Get deprecation metrics"""
        metrics = CompatibilityLayer.get_deprecation_metrics()
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'total_calls': sum(metrics.values())
        })
    
    # Add migration guide endpoint
    @app.route('/api/v2/migration-guide')
    def get_migration_guide():
        """Get API migration guide"""
        guide = CompatibilityLayer.generate_migration_guide()
        return jsonify({
            'status': 'success',
            'guide': guide
        })
    
    logger.info("Compatibility layer registered successfully")