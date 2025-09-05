"""
Service Container
Centralized dependency injection and service management
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Import all services with fallback paths
try:
    from src.services.inventory_analyzer_service import InventoryAnalyzer as InventoryAnalyzerService
    from src.services.sales_forecasting_service import SalesForecastingEngine
    from src.services.capacity_planning_service import CapacityPlanningEngine as CapacityPlanningService
    from src.services.yarn_requirement_service import YarnRequirementCalculatorService as YarnRequirementService
    from src.services.production_scheduler_service import ProductionSchedulerService
    from src.services.manufacturing_supply_chain_service import ManufacturingSupplyChainService
    from src.services.time_phased_mrp_service import TimePhasedMRPService
except ImportError:
    from .inventory_analyzer_service import InventoryAnalyzer as InventoryAnalyzerService
    from .sales_forecasting_service import SalesForecastingEngine
    from .capacity_planning_service import CapacityPlanningEngine as CapacityPlanningService
    from .yarn_requirement_service import YarnRequirementCalculatorService as YarnRequirementService
    from .production_scheduler_service import ProductionSchedulerService
    from .manufacturing_supply_chain_service import ManufacturingSupplyChainService
    from .time_phased_mrp_service import TimePhasedMRPService

# Import data loaders and utilities with fallback paths
try:
    from src.data_loaders.unified_data_loader import UnifiedDataLoader as OptimizedDataLoader
    from src.utils.cache_manager import UnifiedCacheManager
except ImportError:
    try:
        from ..data_loaders.unified_data_loader import UnifiedDataLoader as OptimizedDataLoader
        from ..utils.cache_manager import UnifiedCacheManager
    except ImportError:
        OptimizedDataLoader = None
        UnifiedCacheManager = None

logger = logging.getLogger(__name__)


class Service_Container:
    """Centralized service container with dependency injection"""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(Service_Container, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize service container"""
        if self._initialized:
            return
        
        self._services = {}
        self._factories = {}
        self._config = {}
        self._initialized = True
        self._initialization_time = datetime.now()
        
        # Initialize core services
        self._initialize_core_services()
        
        # Initialize business services
        self._initialize_business_services()
        
        logger.info(f"Service container initialized with {len(self._services)} services")
    
    def _initialize_core_services(self):
        """Initialize core infrastructure services"""
        try:
            # Data loader
            if OptimizedDataLoader is not None:
                self.register('data_loader', OptimizedDataLoader())
                logger.info("Data loader service registered")
            else:
                logger.warning("Data loader service not available")
            
            # Cache manager
            if UnifiedCacheManager is not None:
                self.register('cache', UnifiedCacheManager())
                logger.info("Cache manager service registered")
            else:
                logger.warning("Cache manager service not available")
            
        except Exception as e:
            logger.error(f"Error initializing core services: {e}")
    
    def _initialize_business_services(self):
        """Initialize business logic services"""
        try:
            # Get dependencies
            data_loader = self.get('data_loader') if 'data_loader' in self._services else None
            cache = self.get('cache') if 'cache' in self._services else None
            
            if data_loader is None:
                logger.warning("Skipping business services initialization - data_loader not available")
                return
            
            # Inventory Analyzer Service
            inventory_service = InventoryAnalyzerService(data_loader, cache)
            self.register('inventory', inventory_service)
            self.register('inventory_analyzer', inventory_service)  # Also register with alternate name
            logger.info("Inventory analyzer service registered")
            
            # Sales Forecasting Service
            forecasting_service = SalesForecastingEngine(data_loader, cache)
            self.register('forecasting', forecasting_service)
            self.register('sales_forecasting', forecasting_service)  # Alternate name
            logger.info("Sales forecasting service registered")
            
            # Capacity Planning Service
            capacity_service = CapacityPlanningService(data_loader)
            self.register('capacity', capacity_service)
            self.register('capacity_planning', capacity_service)  # Alternate name
            logger.info("Capacity planning service registered")
            
            # Yarn Requirement Service
            yarn_service = YarnRequirementService(data_loader, cache)
            self.register('yarn', yarn_service)
            self.register('yarn_requirement', yarn_service)  # Alternate name
            logger.info("Yarn requirement service registered")
            
            # Production Scheduler Service
            scheduler_service = ProductionSchedulerService(data_loader, capacity_service)
            self.register('scheduler', scheduler_service)
            self.register('production_scheduler', scheduler_service)  # Alternate name
            logger.info("Production scheduler service registered")
            
            # Manufacturing Supply Chain Service
            supply_chain_service = ManufacturingSupplyChainService(
                inventory_service,
                forecasting_service,
                capacity_service
            )
            self.register('supply_chain', supply_chain_service)
            self.register('manufacturing_supply_chain', supply_chain_service)  # Alternate name
            logger.info("Manufacturing supply chain service registered")
            
            # Time-Phased MRP Service
            mrp_service = TimePhasedMRPService(data_loader, inventory_service)
            self.register('mrp', mrp_service)
            self.register('time_phased_mrp', mrp_service)  # Alternate name
            logger.info("Time-phased MRP service registered")
            
        except Exception as e:
            logger.error(f"Error initializing business services: {e}")
            raise
    
    def get(self, name: str) -> Any:
        """
        Get a service by name
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        if name not in self._services:
            # Check if there's a factory for lazy loading
            if name in self._factories:
                logger.info(f"Lazy loading service: {name}")
                self._services[name] = self._factories[name]()
            else:
                raise KeyError(f"Service '{name}' not found in container")
        
        return self._services[name]
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value
    
    def get_config(self, key: str, default=None):
        """Get configuration value"""
        return self._config.get(key, default)
    
    def initialize_all_services(self):
        """Initialize all services (called by ServiceIntegration)"""
        if not self._services:
            self._initialize_core_services()
            self._initialize_business_services()
    
    def register(self, name: str, service: Any):
        """
        Register a service
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        logger.debug(f"Service '{name}' registered")
    
    def register_factory(self, name: str, factory: Callable):
        """
        Register a service factory for lazy loading
        
        Args:
            name: Service name
            factory: Factory function that returns service instance
        """
        self._factories[name] = factory
        logger.debug(f"Factory for service '{name}' registered")
    
    def has(self, name: str) -> bool:
        """
        Check if service exists
        
        Args:
            name: Service name
            
        Returns:
            True if service exists
        """
        return name in self._services or name in self._factories
    
    def list_services(self) -> list:
        """
        List all registered services
        
        Returns:
            List of service names
        """
        return list(self._services.keys()) + list(self._factories.keys())
    
    def reload_service(self, name: str):
        """
        Reload a service (useful for configuration changes)
        
        Args:
            name: Service name to reload
        """
        if name not in self._services:
            logger.warning(f"Service '{name}' not loaded, cannot reload")
            return
        
        logger.info(f"Reloading service: {name}")
        
        # Special handling for different services
        if name == 'data_loader':
            self._services[name] = OptimizedDataLoader()
        elif name == 'cache':
            # Clear cache before reloading
            old_cache = self._services[name]
            old_cache.clear_all()
            self._services[name] = UnifiedCacheManager()
        else:
            # Re-initialize business services
            self._initialize_business_services()
        
        logger.info(f"Service '{name}' reloaded successfully")
    
    def clear_cache(self):
        """Clear all caches"""
        cache = self.get('cache')
        cache.clear_all()
        logger.info("All caches cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get container status
        
        Returns:
            Status information
        """
        return {
            'initialized': self._initialized,
            'initialization_time': self._initialization_time.isoformat(),
            'services_count': len(self._services),
            'factories_count': len(self._factories),
            'services': self.list_services(),
            'uptime_seconds': (datetime.now() - self._initialization_time).total_seconds()
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all services
        
        Returns:
            Health status for each service
        """
        health = {}
        
        for name in self._services:
            try:
                service = self.get(name)
                
                # Check if service has health_check method
                if hasattr(service, 'health_check'):
                    health[name] = service.health_check()
                else:
                    # Basic check - service exists and is not None
                    health[name] = service is not None
                    
            except Exception as e:
                logger.error(f"Health check failed for service '{name}': {e}")
                health[name] = False
        
        return health
    
    def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down service container")
        
        # Shutdown services in reverse order
        for name in reversed(list(self._services.keys())):
            try:
                service = self._services[name]
                
                # Check if service has shutdown method
                if hasattr(service, 'shutdown'):
                    logger.info(f"Shutting down service: {name}")
                    service.shutdown()
                    
            except Exception as e:
                logger.error(f"Error shutting down service '{name}': {e}")
        
        # Clear services
        self._services.clear()
        self._factories.clear()
        self._initialized = False
        
        logger.info("Service container shutdown complete")


# Global service container instance
services = Service_Container()


def get_service(name: str) -> Any:
    """
    Helper function to get service from global container
    
    Args:
        name: Service name
        
    Returns:
        Service instance
    """
    return services.get(name)


def register_service(name: str, service: Any):
    """
    Helper function to register service in global container
    
    Args:
        name: Service name
        service: Service instance
    """
    services.register(name, service)