"""Adapter to gradually migrate monolith to use dependency-injected services."""

import logging
from typing import Any, Dict, Optional
from flask import Flask
import pandas as pd

from src.infrastructure.container.container import Container, container as global_container
from src.config.feature_flags import FeatureFlags


class MonolithServiceAdapter:
    """
    Adapter that allows the monolith to gradually migrate to using new services.
    This implements the Strangler Fig pattern for gradual migration.
    """
    
    def __init__(self, app: Flask, container: Container = None):
        """Initialize the adapter with Flask app and DI container."""
        self.app = app
        self.container = container or global_container
        self.logger = logging.getLogger(__name__)
        self.feature_flags = FeatureFlags()
        
        # Store original monolith methods for fallback
        self._original_methods = {}
        
        # Initialize the adapter
        self._setup_adapter()
    
    def _setup_adapter(self):
        """Set up the adapter to intercept monolith calls."""
        # Store reference in app config
        self.app.config['MONOLITH_ADAPTER'] = self
        
        # Configure container
        self._configure_container()
        
        # Initialize resources
        try:
            self.container.init_resources()
            self.logger.info("Monolith adapter initialized with DI container")
        except Exception as e:
            self.logger.error(f"Failed to initialize container resources: {e}")
    
    def _configure_container(self):
        """Configure container from Flask app config."""
        # Map Flask config to container
        if 'DATA_BASE_PATH' in self.app.config:
            self.container.config.data.base_path = self.app.config['DATA_BASE_PATH']
        
        if 'REDIS_HOST' in self.app.config:
            self.container.config.redis.host = self.app.config['REDIS_HOST']
            self.container.config.redis.port = self.app.config.get('REDIS_PORT', 6379)
    
    def replace_method(self, target_object: Any, method_name: str, service_name: str, 
                       service_method: str = None, feature_flag: str = None):
        """
        Replace a monolith method with a service method.
        
        Args:
            target_object: The object containing the method to replace
            method_name: Name of the method to replace
            service_name: Name of the service in the container
            service_method: Method name in the service (defaults to method_name)
            feature_flag: Optional feature flag to control migration
        """
        # Store original method
        original_key = f"{target_object.__class__.__name__}.{method_name}"
        self._original_methods[original_key] = getattr(target_object, method_name)
        
        # Create replacement method
        def replacement_method(*args, **kwargs):
            # Check feature flag if provided
            if feature_flag and not self.feature_flags.is_enabled(feature_flag):
                # Use original method
                return self._original_methods[original_key](*args, **kwargs)
            
            try:
                # Get service from container
                service = getattr(self.container, service_name)()
                
                # Get service method
                actual_method_name = service_method or method_name
                service_method_func = getattr(service, actual_method_name)
                
                # Call service method
                self.logger.debug(f"Calling {service_name}.{actual_method_name} instead of {original_key}")
                return service_method_func(*args, **kwargs)
                
            except Exception as e:
                self.logger.error(f"Service call failed, falling back to original: {e}")
                # Fallback to original method
                return self._original_methods[original_key](*args, **kwargs)
        
        # Replace the method
        setattr(target_object, method_name, replacement_method)
        self.logger.info(f"Replaced {original_key} with {service_name}.{service_method or method_name}")
    
    def get_inventory_analyzer(self):
        """Get inventory analyzer service with fallback."""
        if self.feature_flags.is_enabled('use_new_inventory_service'):
            return self.container.inventory_analyzer()
        return None  # Falls back to monolith
    
    def get_forecasting_service(self):
        """Get forecasting service with fallback."""
        if self.feature_flags.is_enabled('use_new_forecasting_service'):
            return self.container.enhanced_forecasting()
        return None  # Falls back to monolith
    
    def get_production_service(self):
        """Get production service with fallback."""
        if self.feature_flags.is_enabled('use_new_production_service'):
            return self.container.production_pipeline()
        return None  # Falls back to monolith
    
    def get_yarn_intelligence(self):
        """Get yarn intelligence service with fallback."""
        if self.feature_flags.is_enabled('use_new_yarn_service'):
            return self.container.yarn_intelligence()
        return None  # Falls back to monolith
    
    def migrate_data_loader(self, monolith_instance: Any):
        """Migrate data loading to use unified data loader."""
        if not self.feature_flags.is_enabled('use_unified_data_loader'):
            return
        
        try:
            # Get unified data loader
            data_loader = self.container.unified_data_loader()
            
            # Replace data loading methods
            def load_yarn_inventory():
                self.logger.debug("Using unified data loader for yarn inventory")
                return data_loader.load_yarn_inventory()
            
            def load_bom_data():
                self.logger.debug("Using unified data loader for BOM data")
                return data_loader.load_bom_data()
            
            def load_production_orders():
                self.logger.debug("Using unified data loader for production orders")
                return data_loader.load_production_orders()
            
            # Apply replacements
            if hasattr(monolith_instance, 'load_yarn_inventory'):
                self._original_methods['load_yarn_inventory'] = monolith_instance.load_yarn_inventory
                monolith_instance.load_yarn_inventory = load_yarn_inventory
            
            if hasattr(monolith_instance, 'load_bom_data'):
                self._original_methods['load_bom_data'] = monolith_instance.load_bom_data
                monolith_instance.load_bom_data = load_bom_data
            
            if hasattr(monolith_instance, 'load_production_orders'):
                self._original_methods['load_production_orders'] = monolith_instance.load_production_orders
                monolith_instance.load_production_orders = load_production_orders
            
            self.logger.info("Data loading migrated to unified data loader")
            
        except Exception as e:
            self.logger.error(f"Failed to migrate data loader: {e}")
    
    def wrap_api_endpoint(self, endpoint_func, service_name: str, service_method: str):
        """
        Wrap an API endpoint to use a service instead of monolith logic.
        
        Args:
            endpoint_func: The original endpoint function
            service_name: Name of the service to use
            service_method: Method to call on the service
        """
        def wrapped_endpoint(*args, **kwargs):
            # Check if we should use the new service
            if self.feature_flags.is_enabled('use_consolidated_api'):
                try:
                    # Get service
                    service = getattr(self.container, service_name)()
                    method = getattr(service, service_method)
                    
                    # Call service method
                    self.logger.debug(f"API using {service_name}.{service_method}")
                    return method(*args, **kwargs)
                    
                except Exception as e:
                    self.logger.error(f"Service call failed in API: {e}")
                    # Fall through to original
            
            # Use original endpoint
            return endpoint_func(*args, **kwargs)
        
        wrapped_endpoint.__name__ = endpoint_func.__name__
        return wrapped_endpoint
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get the current migration status."""
        return {
            'migrated_methods': len(self._original_methods),
            'feature_flags': {
                'use_new_inventory_service': self.feature_flags.is_enabled('use_new_inventory_service'),
                'use_new_forecasting_service': self.feature_flags.is_enabled('use_new_forecasting_service'),
                'use_new_production_service': self.feature_flags.is_enabled('use_new_production_service'),
                'use_new_yarn_service': self.feature_flags.is_enabled('use_new_yarn_service'),
                'use_unified_data_loader': self.feature_flags.is_enabled('use_unified_data_loader'),
                'use_consolidated_api': self.feature_flags.is_enabled('use_consolidated_api')
            },
            'services_available': {
                'inventory_analyzer': hasattr(self.container, 'inventory_analyzer'),
                'production_pipeline': hasattr(self.container, 'production_pipeline'),
                'enhanced_forecasting': hasattr(self.container, 'enhanced_forecasting'),
                'yarn_intelligence': hasattr(self.container, 'yarn_intelligence'),
                'unified_data_loader': hasattr(self.container, 'unified_data_loader')
            }
        }
    
    def rollback(self, target_object: Any = None, method_name: str = None):
        """
        Rollback to original methods.
        
        Args:
            target_object: Specific object to rollback (None for all)
            method_name: Specific method to rollback (None for all)
        """
        if target_object and method_name:
            # Rollback specific method
            key = f"{target_object.__class__.__name__}.{method_name}"
            if key in self._original_methods:
                setattr(target_object, method_name, self._original_methods[key])
                self.logger.info(f"Rolled back {key}")
        else:
            # Rollback all would require storing object references
            self.logger.warning("Full rollback requires application restart")
    
    def enable_gradual_migration(self, percentage: int = 10):
        """
        Enable gradual migration for a percentage of requests.
        
        Args:
            percentage: Percentage of requests to use new services (0-100)
        """
        # Enable feature flags with percentage rollout
        self.feature_flags.enable('use_new_inventory_service', percentage)
        self.feature_flags.enable('use_new_forecasting_service', percentage)
        self.feature_flags.enable('use_new_production_service', percentage)
        self.feature_flags.enable('use_new_yarn_service', percentage)
        self.feature_flags.enable('use_unified_data_loader', percentage)
        
        self.logger.info(f"Gradual migration enabled for {percentage}% of requests")