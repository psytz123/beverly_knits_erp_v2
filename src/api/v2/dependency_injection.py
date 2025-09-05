#!/usr/bin/env python3
"""
Dependency Injection Container - Phase 4
Manages service dependencies and injection for API v2
"""

from typing import Any, Dict, Type, Optional, Callable
from functools import wraps
import logging
from flask import g

logger = logging.getLogger(__name__)

class ServiceContainer:
    """Dependency injection container for managing services"""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
        self._factories = {}
        
    def register(self, name: str, factory: Callable, singleton: bool = True):
        """
        Register a service factory
        
        Args:
            name: Service name
            factory: Factory function to create the service
            singleton: Whether to create a single instance or new instances
        """
        self._factories[name] = factory
        if singleton:
            self._singletons[name] = None
        logger.info(f"Registered service: {name} (singleton={singleton})")
    
    def get(self, name: str) -> Any:
        """
        Get a service instance
        
        Args:
            name: Service name
            
        Returns:
            Service instance
        """
        if name not in self._factories:
            raise ValueError(f"Service '{name}' not registered")
        
        # Check if it's a singleton and already created
        if name in self._singletons:
            if self._singletons[name] is None:
                # Create the singleton instance
                self._singletons[name] = self._factories[name]()
            return self._singletons[name]
        
        # Create new instance for non-singletons
        return self._factories[name]()
    
    def reset(self):
        """Reset all singleton instances"""
        for name in self._singletons:
            self._singletons[name] = None
        logger.info("Service container reset")

# Global container instance
container = ServiceContainer()

def inject(**dependencies):
    """
    Decorator for dependency injection into Flask routes
    
    Usage:
        @inject(inventory_service='inventory_service')
        def my_route(inventory_service):
            return inventory_service.get_data()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject dependencies
            for param_name, service_name in dependencies.items():
                if param_name not in kwargs:
                    kwargs[param_name] = container.get(service_name)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def register_services():
    """Register all services with the container"""
    
    # Import services lazily to avoid circular imports
    def create_inventory_service():
        try:
            from src.services.inventory_service import InventoryService
            from src.core.beverly_comprehensive_erp import InventoryAnalyzer
            analyzer = InventoryAnalyzer()
            return InventoryService(analyzer)
        except ImportError:
            # Return mock service if imports fail
            logger.warning("Could not import InventoryService, using mock")
            class MockInventoryService:
                def get_inventory(self): return {'items': []}
                def get_shortages(self): return {'shortages': []}
            return MockInventoryService()
    
    def create_production_service():
        try:
            from src.services.production_service import ProductionService
            return ProductionService()
        except ImportError:
            # Return mock service if imports fail
            logger.warning("Could not import ProductionService, using mock")
            class MockProductionService:
                def get_orders(self): return {'orders': []}
                def get_schedule(self): return {'schedule': {}}
            return MockProductionService()
    
    
    def create_forecasting_service():
        try:
            from src.services.forecasting_service import ForecastingService
            return ForecastingService()
        except ImportError:
            logger.warning("Could not import ForecastingService, using mock")
            class MockForecastingService:
                def get_forecast(self): return {'forecast': {}}
                def run_ensemble(self): return {'ensemble': {}}
            return MockForecastingService()
    
    def create_yarn_service():
        try:
            from src.services.yarn_service import YarnService
            return YarnService()
        except ImportError:
            logger.warning("Could not import YarnService, using mock")
            class MockYarnService:
                def get_yarns(self): return {'yarns': []}
                def get_shortages(self): return {'shortages': []}
            return MockYarnService()
    
    # Register services
    container.register('inventory_service', create_inventory_service)
    container.register('production_service', create_production_service)
    container.register('forecasting_service', create_forecasting_service)
    container.register('yarn_service', create_yarn_service)
    
    logger.info("All services registered with container")

# Service wrapper classes for better organization
class BaseService:
    """Base service class"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

class InventoryService(BaseService):
    """Inventory management service"""
    
    def __init__(self, inventory_analyzer):
        super().__init__()
        self.analyzer = inventory_analyzer
    
    def get_inventory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get inventory based on parameters"""
        view = params.get('view', 'summary')
        
        if view == 'yarn':
            return self.analyzer.get_yarn_inventory()
        elif view == 'shortage':
            return self.analyzer.get_shortage_analysis()
        elif view == 'planning':
            return self.analyzer.get_planning_balance_analysis()
        else:
            return self.analyzer.get_inventory_summary()
    
    def get_shortages(self) -> list:
        """Get current shortages"""
        return self.analyzer.detect_shortages()

class ProductionService(BaseService):
    """Production management service"""
    
    def __init__(self, get_planning_func, get_status_func):
        super().__init__()
        self.get_planning = get_planning_func
        self.get_status = get_status_func
    
    def get_production_data(self, view: str) -> Dict[str, Any]:
        """Get production data based on view"""
        if view == 'planning':
            return self.get_planning()
        else:
            return self.get_status()
    
    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new production order"""
        # Implement order creation logic
        return {
            'id': 'new_order_id',
            'status': 'created',
            **order_data
        }

class ForecastingService(BaseService):
    """Forecasting service"""
    
    def __init__(self, get_forecast_func, trigger_retrain_func):
        super().__init__()
        self.get_forecast = get_forecast_func
        self.trigger_retrain = trigger_retrain_func
    
    def get_forecast_data(self, model: str, horizon: int) -> Dict[str, Any]:
        """Get forecast data"""
        if self.get_forecast:
            return self.get_forecast(model=model, horizon=horizon)
        return {'error': 'Forecasting not available'}
    
    def retrain_model(self, model: str, force: bool = False) -> Dict[str, Any]:
        """Trigger model retraining"""
        if self.trigger_retrain:
            return self.trigger_retrain(model, force=force)
        return {'error': 'Retraining not available'}

class YarnService(BaseService):
    """Yarn management service"""
    
    def __init__(self, get_intelligence_func):
        super().__init__()
        self.get_intelligence = get_intelligence_func
    
    def get_yarn_data(self, yarn_id: Optional[str] = None) -> Dict[str, Any]:
        """Get yarn intelligence data"""
        if self.get_intelligence:
            return self.get_intelligence(yarn_id)
        return {'error': 'Yarn intelligence not available'}

# Service factory functions
def create_services():
    """Create service instances for the application"""
    services = {}
    
    try:
        # Create inventory service
        from src.core.beverly_comprehensive_erp import inventory_analyzer
        services['inventory'] = InventoryService(inventory_analyzer)
    except ImportError as e:
        logger.error(f"Failed to create inventory service: {e}")
    
    try:
        # Create production service
        from src.core.beverly_comprehensive_erp import (
            get_production_planning,
            get_production_status
        )
        services['production'] = ProductionService(
            get_production_planning,
            get_production_status
        )
    except ImportError as e:
        logger.error(f"Failed to create production service: {e}")
    
    try:
        # Create forecasting service
        from src.forecasting.enhanced_forecasting_engine import (
            get_forecast,
            trigger_retraining
        )
        services['forecasting'] = ForecastingService(
            get_forecast,
            trigger_retraining
        )
    except ImportError as e:
        logger.warning(f"Forecasting service not available: {e}")
    
    try:
        # Create yarn service
        from src.yarn_intelligence.yarn_intelligence_enhanced import (
            get_yarn_intelligence
        )
        services['yarn'] = YarnService(get_yarn_intelligence)
    except ImportError as e:
        logger.warning(f"Yarn service not available: {e}")
    
    return services

# Flask integration helpers
def init_dependency_injection(app):
    """Initialize dependency injection for Flask app"""
    
    @app.before_request
    def before_request():
        """Create service instances for each request"""
        if not hasattr(g, 'services'):
            g.services = create_services()
    
    @app.teardown_request
    def teardown_request(exception):
        """Clean up services after request"""
        if hasattr(g, 'services'):
            delattr(g, 'services')
    
    # Register services with container
    register_services()
    
    logger.info("Dependency injection initialized")

# Export main components
__all__ = [
    'container',
    'inject',
    'register_services',
    'init_dependency_injection',
    'InventoryService',
    'ProductionService',
    'ForecastingService',
    'YarnService'
]