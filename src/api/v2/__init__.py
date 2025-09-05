"""
V2 API Package - Consolidated endpoints for Beverly Knits ERP
"""

from flask import Flask
import logging
from typing import Optional

# Import blueprints
from .inventory import inventory_v2_bp, initialize_inventory_handler
from .yarn import yarn_v2_bp, initialize_yarn_handler  
from .production import production_v2_bp, initialize_production_handler

logger = logging.getLogger(__name__)

# Export blueprints
__all__ = [
    'inventory_v2_bp',
    'yarn_v2_bp',
    'production_v2_bp',
    'register_v2_endpoints',
    'initialize_v2_handlers'
]


def register_v2_endpoints(app: Flask):
    """
    Register all v2 API blueprints with the Flask application
    
    Args:
        app: Flask application instance
    """
    try:
        # Register blueprints
        app.register_blueprint(inventory_v2_bp)
        app.register_blueprint(yarn_v2_bp)
        app.register_blueprint(production_v2_bp)
        
        logger.info("V2 API endpoints registered successfully")
        
        # Log registered routes
        v2_routes = [rule.rule for rule in app.url_map.iter_rules() if '/api/v2/' in rule.rule]
        logger.info(f"V2 routes registered: {v2_routes}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register v2 endpoints: {e}")
        return False


def initialize_v2_handlers(analyzer=None, api_client=None, production_engine=None):
    """
    Initialize all v2 API handlers with required dependencies
    
    Args:
        analyzer: Inventory analyzer instance
        api_client: eFab API client instance  
        production_engine: Production planning engine instance
    """
    try:
        # Initialize handlers
        initialize_inventory_handler(analyzer, api_client)
        initialize_yarn_handler(analyzer, api_client)
        initialize_production_handler(analyzer, api_client, production_engine)
        
        logger.info("V2 API handlers initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize v2 handlers: {e}")
        return False