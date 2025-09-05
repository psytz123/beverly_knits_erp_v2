"""Flask integration for dependency injection container."""

import logging
from flask import Flask, current_app
from functools import wraps
from typing import Any, Callable
from dependency_injector.wiring import Provide

from .container import Container, container as global_container


class FlaskContainerIntegration:
    """Integration layer between Flask and DI container."""
    
    def __init__(self, app: Flask = None, container_instance: Container = None):
        """Initialize Flask integration."""
        self.container = container_instance or global_container
        self.logger = logging.getLogger(__name__)
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask application with DI container."""
        # Store container in app config
        app.config['DI_CONTAINER'] = self.container
        
        # Configure container from app config
        self._configure_container(app)
        
        # Wire container to Flask app
        self.container.wire(modules=[__name__])
        
        # Initialize resources on first request
        @app.before_first_request
        def init_resources():
            try:
                self.container.init_resources()
                self.logger.info("DI Container resources initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize resources: {e}")
        
        # Cleanup on teardown
        @app.teardown_appcontext
        def cleanup_resources(error=None):
            if error:
                self.logger.error(f"Request error: {error}")
        
        # Add container to app extensions
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['di_container'] = self.container
        
        self.logger.info("Flask DI Container integration initialized")
    
    def _configure_container(self, app: Flask):
        """Configure container from Flask config."""
        # Map Flask config to container config
        config_mapping = {
            'DATA_BASE_PATH': 'data.base_path',
            'CACHE_DIR': 'cache.dir',
            'CACHE_REDIS_ENABLED': 'cache.redis_enabled',
            'REDIS_HOST': 'redis.host',
            'REDIS_PORT': 'redis.port',
            'CACHE_DEFAULT_TTL': 'cache.default_ttl',
            'ML_MODELS_PATH': 'ml.models_path',
            'EFAB_API_URL': 'efab.api_url',
            'EFAB_API_KEY': 'efab.api_key',
            'USE_PARALLEL_LOADING': 'data.use_parallel',
            'MAX_WORKERS': 'data.max_workers',
            'WARM_CACHE_ON_STARTUP': 'cache.warm_on_startup'
        }
        
        for flask_key, container_key in config_mapping.items():
            if flask_key in app.config:
                # Set nested config values
                keys = container_key.split('.')
                current = self.container.config
                for key in keys[:-1]:
                    if not hasattr(current, key):
                        setattr(current, key, {})
                    current = getattr(current, key)
                setattr(current, keys[-1], app.config[flask_key])


def inject_service(service_name: str):
    """Decorator to inject a service into a Flask route."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get container from current app
            container = current_app.config.get('DI_CONTAINER')
            if not container:
                raise RuntimeError("DI Container not initialized in Flask app")
            
            # Get service from container
            service = getattr(container, service_name)()
            
            # Add service to kwargs
            kwargs[service_name] = service
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_service(service_name: str) -> Any:
    """Get a service from the container in Flask context."""
    container = current_app.config.get('DI_CONTAINER')
    if not container:
        raise RuntimeError("DI Container not initialized in Flask app")
    
    service_provider = getattr(container, service_name, None)
    if not service_provider:
        raise ValueError(f"Service '{service_name}' not found in container")
    
    return service_provider()


# Convenience functions for common services
def get_inventory_service():
    """Get inventory analyzer service."""
    return get_service('inventory_analyzer')


def get_forecasting_service():
    """Get sales forecasting service."""
    return get_service('sales_forecasting')


def get_production_service():
    """Get production pipeline service."""
    return get_service('production_pipeline')


def get_yarn_service():
    """Get yarn intelligence service."""
    return get_service('yarn_intelligence')


def get_data_loader():
    """Get unified data loader."""
    return get_service('unified_data_loader')


def get_cache_manager():
    """Get cache manager."""
    return get_service('cache_manager')


def get_service_manager():
    """Get ERP service manager."""
    return get_service('service_manager')