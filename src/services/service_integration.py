"""
Service Integration Module
Wires up extracted services to the main monolith application
Created: 2025-09-05
Purpose: Connect 7+ existing services to reduce monolith from 18,076 lines
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Import the service container
try:
    from src.services.service_container import Service_Container as ServiceContainer
except ImportError:
    try:
        from .service_container import Service_Container as ServiceContainer
    except ImportError as e:
        print(f"[ERROR] ServiceContainer not available: {e}")
        ServiceContainer = None

# Import individual services for direct access if needed
try:
    from src.services.inventory_analyzer_service import InventoryAnalyzer
    from src.services.sales_forecasting_service import SalesForecastingEngine
    from src.services.capacity_planning_service import CapacityPlanningEngine
    from src.services.yarn_requirement_service import YarnRequirementCalculatorService
    from src.services.production_scheduler_service import ProductionSchedulerService
    from src.services.manufacturing_supply_chain_service import ManufacturingSupplyChainService
    from src.services.time_phased_mrp_service import TimePhasedMRPService
    SERVICES_AVAILABLE = True
except ImportError:
    try:
        from .inventory_analyzer_service import InventoryAnalyzer
        from .sales_forecasting_service import SalesForecastingEngine
        from .capacity_planning_service import CapacityPlanningEngine
        from .yarn_requirement_service import YarnRequirementCalculatorService
        from .production_scheduler_service import ProductionSchedulerService
        from .manufacturing_supply_chain_service import ManufacturingSupplyChainService
        from .time_phased_mrp_service import TimePhasedMRPService
        SERVICES_AVAILABLE = True
    except ImportError as e:
        SERVICES_AVAILABLE = False
        print(f"[WARNING] Some services not available: {e}")

# Import data loaders
try:
    from src.data_loaders.unified_data_loader import UnifiedDataLoader as OptimizedDataLoader
    from src.utils.cache_manager import UnifiedCacheManager
    DATA_LOADER_AVAILABLE = True
except ImportError:
    try:
        from ..data_loaders.unified_data_loader import UnifiedDataLoader as OptimizedDataLoader
        from ..utils.cache_manager import UnifiedCacheManager
        DATA_LOADER_AVAILABLE = True
    except ImportError:
        DATA_LOADER_AVAILABLE = False
        print("[WARNING] Data loaders not available")

logger = logging.getLogger(__name__)


class ServiceIntegration:
    """
    Integrates extracted services into the main ERP application
    Reduces monolith size by delegating to service modules
    """
    
    def __init__(self, data_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize service integration
        
        Args:
            data_path: Path to data directory
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.config = config or {}
        self.services = {}
        self.container = None
        self._initialized = False
        
        # Initialize services
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize all services using the service container"""
        try:
            # Get or create service container
            self.container = ServiceContainer()
            
            # Configure container with data path if provided
            if self.data_path:
                self.container.set_config('data_path', self.data_path)
            
            # Configure additional settings
            for key, value in self.config.items():
                self.container.set_config(key, value)
            
            # Initialize services through container
            self.container.initialize_all_services()
            
            # Store service references for easy access
            self.services = {
                'inventory': self.container.get('inventory_analyzer'),
                'forecasting': self.container.get('sales_forecasting'),
                'capacity': self.container.get('capacity_planning'),
                'yarn': self.container.get('yarn_requirement'),
                'scheduler': self.container.get('production_scheduler'),
                'supply_chain': self.container.get('manufacturing_supply_chain'),
                'mrp': self.container.get('time_phased_mrp'),
                'data_loader': self.container.get('data_loader'),
                'cache': self.container.get('cache_manager')
            }
            
            self._initialized = True
            logger.info(f"[OK] Service integration initialized with {len(self.services)} services")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            self._initialized = False
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a specific service by name
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance or None
        """
        return self.services.get(service_name)
    
    def is_initialized(self) -> bool:
        """Check if services are initialized"""
        return self._initialized
    
    # Delegation methods for backward compatibility with monolith
    
    def analyze_inventory(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate inventory analysis to inventory service
        Replaces monolith's inventory_analyzer logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        inventory_service = self.get_service('inventory')
        if inventory_service:
            return inventory_service.analyze_inventory(**kwargs)
        else:
            raise RuntimeError("Inventory service not available")
    
    def forecast_demand(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate demand forecasting to forecasting service
        Replaces monolith's sales_forecasting_engine logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        forecasting_service = self.get_service('forecasting')
        if forecasting_service:
            return forecasting_service.forecast(**kwargs)
        else:
            raise RuntimeError("Forecasting service not available")
    
    def plan_capacity(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate capacity planning to capacity service
        Replaces monolith's capacity_planning_engine logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        capacity_service = self.get_service('capacity')
        if capacity_service:
            return capacity_service.plan_capacity(**kwargs)
        else:
            raise RuntimeError("Capacity planning service not available")
    
    def calculate_yarn_requirements(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate yarn requirement calculation to yarn service
        Replaces monolith's yarn_requirement_calculator logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        yarn_service = self.get_service('yarn')
        if yarn_service:
            return yarn_service.calculate_requirements(**kwargs)
        else:
            raise RuntimeError("Yarn requirement service not available")
    
    def schedule_production(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate production scheduling to scheduler service
        Replaces monolith's production_scheduler logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        scheduler_service = self.get_service('scheduler')
        if scheduler_service:
            return scheduler_service.schedule(**kwargs)
        else:
            raise RuntimeError("Production scheduler service not available")
    
    def optimize_supply_chain(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate supply chain optimization to supply chain service
        Replaces monolith's manufacturing_supply_chain logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        supply_chain_service = self.get_service('supply_chain')
        if supply_chain_service:
            return supply_chain_service.optimize(**kwargs)
        else:
            raise RuntimeError("Supply chain service not available")
    
    def calculate_mrp(self, **kwargs) -> Dict[str, Any]:
        """
        Delegate MRP calculation to MRP service
        Replaces monolith's time_phased_mrp logic
        """
        if not self._initialized:
            raise RuntimeError("Services not initialized")
        
        mrp_service = self.get_service('mrp')
        if mrp_service:
            return mrp_service.calculate(**kwargs)
        else:
            raise RuntimeError("MRP service not available")
    
    def get_data_loader(self) -> Optional[Any]:
        """Get the optimized data loader"""
        return self.get_service('data_loader')
    
    def get_cache_manager(self) -> Optional[Any]:
        """Get the cache manager"""
        return self.get_service('cache')
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all services
        
        Returns:
            Health status dictionary
        """
        health = {
            'initialized': self._initialized,
            'services': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, service in self.services.items():
            try:
                if service is not None:
                    # Check if service has health_check method
                    if hasattr(service, 'health_check'):
                        health['services'][name] = service.health_check()
                    else:
                        health['services'][name] = {'status': 'OK', 'available': True}
                else:
                    health['services'][name] = {'status': 'NOT_LOADED', 'available': False}
            except Exception as e:
                health['services'][name] = {'status': 'ERROR', 'error': str(e), 'available': False}
        
        # Overall health status
        all_healthy = all(
            s.get('available', False) for s in health['services'].values()
        )
        health['overall_status'] = 'HEALTHY' if all_healthy else 'DEGRADED'
        
        return health


# Singleton instance for easy import
_service_integration = None

def get_service_integration(data_path: str = None, config: Dict[str, Any] = None) -> ServiceIntegration:
    """
    Get or create the service integration singleton
    
    Args:
        data_path: Path to data directory
        config: Configuration dictionary
        
    Returns:
        ServiceIntegration instance
    """
    global _service_integration
    
    if _service_integration is None:
        _service_integration = ServiceIntegration(data_path, config)
    
    return _service_integration


def integrate_with_monolith(monolith_instance):
    """
    Integrate services with existing monolith instance
    This function patches the monolith to use services instead of embedded logic
    
    Args:
        monolith_instance: Instance of ManufacturingSupplyChainAI
    """
    try:
        # Get service integration
        integration = get_service_integration(
            data_path=getattr(monolith_instance, 'data_path', None)
        )
        
        # Store integration reference in monolith
        monolith_instance.service_integration = integration
        
        # Replace monolith methods with service delegates
        # This allows gradual migration without breaking existing code
        
        # Inventory analysis
        if hasattr(monolith_instance, 'analyze_inventory'):
            monolith_instance.analyze_inventory = integration.analyze_inventory
            logger.info("[OK] Replaced monolith inventory analysis with service")
        
        # Demand forecasting
        if hasattr(monolith_instance, 'forecast_demand'):
            monolith_instance.forecast_demand = integration.forecast_demand
            logger.info("[OK] Replaced monolith demand forecasting with service")
        
        # Capacity planning
        if hasattr(monolith_instance, 'plan_capacity'):
            monolith_instance.plan_capacity = integration.plan_capacity
            logger.info("[OK] Replaced monolith capacity planning with service")
        
        # Yarn requirements
        if hasattr(monolith_instance, 'calculate_yarn_requirements'):
            monolith_instance.calculate_yarn_requirements = integration.calculate_yarn_requirements
            logger.info("[OK] Replaced monolith yarn calculation with service")
        
        # Production scheduling
        if hasattr(monolith_instance, 'schedule_production'):
            monolith_instance.schedule_production = integration.schedule_production
            logger.info("[OK] Replaced monolith production scheduling with service")
        
        # Supply chain optimization
        if hasattr(monolith_instance, 'optimize_supply_chain'):
            monolith_instance.optimize_supply_chain = integration.optimize_supply_chain
            logger.info("[OK] Replaced monolith supply chain optimization with service")
        
        # MRP calculation
        if hasattr(monolith_instance, 'calculate_mrp'):
            monolith_instance.calculate_mrp = integration.calculate_mrp
            logger.info("[OK] Replaced monolith MRP calculation with service")
        
        logger.info(f"[SUCCESS] Integrated {len(integration.services)} services with monolith")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate services with monolith: {e}")
        return False