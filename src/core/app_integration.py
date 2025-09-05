"""Integration module to wire dependency injection into the main ERP application."""

import logging
import sys
from pathlib import Path
from flask import Flask

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.infrastructure.container.container import Container, configure_container
from src.infrastructure.container.flask_integration import FlaskContainerIntegration
from src.infrastructure.adapters.monolith_adapter import MonolithServiceAdapter
from src.api.v2.routes import api_v2
from src.config.feature_flags import FeatureFlags


logger = logging.getLogger(__name__)


def integrate_with_monolith(app: Flask, monolith_instance=None):
    """
    Integrate the new service architecture with the existing monolith.
    This is the main entry point for the migration.
    
    Args:
        app: Flask application instance
        monolith_instance: The main monolith class instance (e.g., InventoryAnalyzer)
    """
    logger.info("Starting integration of new service architecture...")
    
    # Step 1: Configure and initialize DI container
    container = configure_container()
    
    # Step 2: Integrate container with Flask
    flask_integration = FlaskContainerIntegration(app, container)
    
    # Step 3: Set up monolith adapter for gradual migration
    adapter = MonolithServiceAdapter(app, container)
    
    # Step 4: Register new API v2 routes
    app.register_blueprint(api_v2)
    logger.info("Registered API v2 routes at /api/v2/*")
    
    # Step 5: Apply gradual migration if monolith instance provided
    if monolith_instance:
        _apply_service_replacements(adapter, monolith_instance)
    
    # Step 6: Set up feature flags for controlled rollout
    _configure_feature_flags(app)
    
    # Step 7: Add middleware for deprecated endpoint handling
    _setup_deprecated_endpoint_middleware(app, adapter)
    
    logger.info("Integration complete - services ready for gradual migration")
    
    return adapter


def _apply_service_replacements(adapter: MonolithServiceAdapter, monolith_instance):
    """Apply service replacements to monolith methods."""
    
    # Replace inventory methods
    if hasattr(monolith_instance, 'get_inventory_summary'):
        adapter.replace_method(
            target_object=monolith_instance,
            method_name='get_inventory_summary',
            service_name='inventory_analyzer',
            feature_flag='use_new_inventory_service'
        )
    
    if hasattr(monolith_instance, 'detect_shortages'):
        adapter.replace_method(
            target_object=monolith_instance,
            method_name='detect_shortages',
            service_name='inventory_analyzer',
            feature_flag='use_new_inventory_service'
        )
    
    # Replace forecasting methods
    if hasattr(monolith_instance, 'generate_forecast'):
        adapter.replace_method(
            target_object=monolith_instance,
            method_name='generate_forecast',
            service_name='enhanced_forecasting',
            feature_flag='use_new_forecasting_service'
        )
    
    # Replace production methods
    if hasattr(monolith_instance, 'create_production_plan'):
        adapter.replace_method(
            target_object=monolith_instance,
            method_name='create_production_plan',
            service_name='six_phase_planning',
            feature_flag='use_new_production_service'
        )
    
    # Migrate data loading
    adapter.migrate_data_loader(monolith_instance)
    
    logger.info(f"Applied service replacements to {monolith_instance.__class__.__name__}")


def _configure_feature_flags(app: Flask):
    """Configure feature flags for gradual rollout."""
    
    # Get or create feature flags instance
    flags = FeatureFlags()
    
    # Set initial feature flag states (all disabled by default for safety)
    initial_flags = {
        'use_new_inventory_service': False,
        'use_new_forecasting_service': False,
        'use_new_production_service': False,
        'use_new_yarn_service': False,
        'use_unified_data_loader': False,
        'use_consolidated_api': False,
        'api_consolidation_enabled': False,
        'redirect_deprecated_apis': False
    }
    
    for flag_name, enabled in initial_flags.items():
        flags.set_flag(flag_name, enabled)
    
    # Store in app config
    app.config['FEATURE_FLAGS'] = flags
    
    logger.info(f"Configured {len(initial_flags)} feature flags")


def _setup_deprecated_endpoint_middleware(app: Flask, adapter: MonolithServiceAdapter):
    """Set up middleware to handle deprecated endpoints."""
    
    # Mapping of deprecated endpoints to new v2 endpoints
    deprecated_mappings = {
        '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
        '/api/yarn-data': '/api/v2/inventory?view=yarn',
        '/api/inventory-intelligence-enhanced': '/api/v2/inventory?analysis=intelligence',
        '/api/real-time-inventory-dashboard': '/api/v2/inventory?realtime=true',
        '/api/emergency-shortage-dashboard': '/api/v2/inventory?view=shortage',
        '/api/production-planning': '/api/v2/production?view=planning',
        '/api/production-status': '/api/v2/production?view=status',
        '/api/production-pipeline': '/api/v2/production?view=pipeline',
        '/api/production-recommendations-ml': '/api/v2/production?view=recommendations',
        '/api/machine-assignment-suggestions': '/api/v2/production?view=machines',
        '/api/ml-forecasting': '/api/v2/forecast',
        '/api/ml-forecast-detailed': '/api/v2/forecast?detail=full',
        '/api/yarn-intelligence': '/api/v2/yarn?action=status',
        '/api/yarn-substitution-intelligent': '/api/v2/yarn?action=substitution',
        '/api/comprehensive-kpis': '/api/v2/kpis',
        '/api/inventory-netting': '/api/v2/netting'
    }
    
    @app.before_request
    def handle_deprecated_endpoints():
        """Middleware to redirect deprecated endpoints if feature flag enabled."""
        from flask import request, redirect
        
        flags = app.config.get('FEATURE_FLAGS')
        if not flags or not flags.is_enabled('redirect_deprecated_apis'):
            return None
        
        # Check if current path is deprecated
        if request.path in deprecated_mappings:
            new_path = deprecated_mappings[request.path]
            
            # Preserve query parameters
            if request.query_string:
                new_path += '&' + request.query_string.decode('utf-8')
            
            logger.warning(f"Deprecated endpoint accessed: {request.path} -> {new_path}")
            
            # Return redirect
            return redirect(new_path, code=301)
        
        return None
    
    logger.info(f"Set up middleware for {len(deprecated_mappings)} deprecated endpoints")


def enable_gradual_migration(app: Flask, percentage: int = 10):
    """
    Enable gradual migration for a percentage of traffic.
    
    Args:
        app: Flask application
        percentage: Percentage of traffic to migrate (0-100)
    """
    adapter = app.config.get('MONOLITH_ADAPTER')
    if adapter:
        adapter.enable_gradual_migration(percentage)
        logger.info(f"Enabled gradual migration for {percentage}% of traffic")
    else:
        logger.error("No adapter found - run integrate_with_monolith first")


def get_migration_status(app: Flask) -> dict:
    """Get current migration status."""
    adapter = app.config.get('MONOLITH_ADAPTER')
    if adapter:
        return adapter.get_migration_status()
    return {'error': 'No adapter found'}


def rollback_migration(app: Flask):
    """Rollback the migration to use original monolith."""
    flags = app.config.get('FEATURE_FLAGS')
    if flags:
        # Disable all migration flags
        flags.set_flag('use_new_inventory_service', False)
        flags.set_flag('use_new_forecasting_service', False)
        flags.set_flag('use_new_production_service', False)
        flags.set_flag('use_new_yarn_service', False)
        flags.set_flag('use_unified_data_loader', False)
        flags.set_flag('use_consolidated_api', False)
        flags.set_flag('redirect_deprecated_apis', False)
        
        logger.info("Rolled back all feature flags - using monolith")
    else:
        logger.error("No feature flags found")