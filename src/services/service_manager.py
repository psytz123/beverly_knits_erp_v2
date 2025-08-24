#!/usr/bin/env python3
"""
Beverly Knits ERP - Service Manager
Central orchestration for all extracted services
Implements dependency injection and service lifecycle management
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import extracted services
from services.inventory_analyzer_service import InventoryAnalyzerService, InventoryConfig
from services.sales_forecasting_service import SalesForecastingService, ForecastConfig
from services.capacity_planning_service import CapacityPlanningService, CapacityConfig
from services.inventory_pipeline_service import InventoryManagementPipelineService, PipelineConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Central service manager for Beverly Knits ERP
    Manages initialization, dependency injection, and lifecycle of all services
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize service manager with optional configuration
        
        Args:
            config: Optional configuration dictionary for services
        """
        self.config = config or {}
        self.services = {}
        self.initialized = False
        
        logger.info("ServiceManager initializing...")
        
        # Initialize all services
        self._initialize_services()
        
        logger.info(f"ServiceManager ready with {len(self.services)} services")
    
    def _initialize_services(self):
        """Initialize all available services with proper configuration"""
        
        # Initialize Inventory Analyzer Service
        try:
            inventory_config = InventoryConfig(
                safety_stock_multiplier=self.config.get('inventory', {}).get('safety_stock_multiplier', 1.5),
                lead_time_days=self.config.get('inventory', {}).get('lead_time_days', 30)
            )
            self.services['inventory'] = InventoryAnalyzerService(inventory_config)
            logger.info("  ✓ InventoryAnalyzerService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize InventoryAnalyzerService: {e}")
        
        # Initialize Sales Forecasting Service
        try:
            forecast_config = ForecastConfig(
                forecast_horizon=self.config.get('forecast', {}).get('horizon_days', 90),
                target_accuracy=self.config.get('forecast', {}).get('target_accuracy', 0.85)
            )
            self.services['forecasting'] = SalesForecastingService(forecast_config)
            logger.info("  ✓ SalesForecastingService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize SalesForecastingService: {e}")
        
        # Initialize Capacity Planning Service
        try:
            capacity_config = CapacityConfig(
                bottleneck_threshold=self.config.get('capacity', {}).get('bottleneck_threshold', 0.85),
                critical_threshold=self.config.get('capacity', {}).get('critical_threshold', 0.95)
            )
            self.services['capacity'] = CapacityPlanningService(capacity_config)
            logger.info("  ✓ CapacityPlanningService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize CapacityPlanningService: {e}")
        
        # Initialize Inventory Pipeline Service
        try:
            pipeline_config = PipelineConfig(
                waste_factor=self.config.get('pipeline', {}).get('waste_factor', 1.2),
                growth_forecast=self.config.get('pipeline', {}).get('growth_forecast', 1.1),
                critical_threshold=self.config.get('pipeline', {}).get('critical_threshold', 5)
            )
            pipeline_service = InventoryManagementPipelineService(pipeline_config)
            # Set dependencies
            pipeline_service.set_dependencies(
                inventory_analyzer=self.services.get('inventory'),
                supply_chain_ai=None  # Will be set when available
            )
            self.services['pipeline'] = pipeline_service
            logger.info("  ✓ InventoryManagementPipelineService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize InventoryManagementPipelineService: {e}")
        
        # TODO: Add more services as they are extracted
        # - YarnRequirementCalculatorService
        # - MultiStageInventoryTrackerService
        
        self.initialized = True
    
    def get_service(self, service_name: str):
        """
        Get a specific service by name
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance or None if not found
        """
        return self.services.get(service_name)
    
    def get_inventory_analyzer(self) -> Optional[InventoryAnalyzerService]:
        """Get inventory analyzer service"""
        return self.services.get('inventory')
    
    def get_sales_forecaster(self) -> Optional[SalesForecastingService]:
        """Get sales forecasting service"""
        return self.services.get('forecasting')
    
    def get_capacity_planner(self) -> Optional[CapacityPlanningService]:
        """Get capacity planning service"""
        return self.services.get('capacity')
    
    def get_inventory_pipeline(self) -> Optional[InventoryManagementPipelineService]:
        """Get inventory management pipeline service"""
        return self.services.get('pipeline')
    
    def analyze_inventory(self, current_inventory, forecast):
        """
        Convenience method for inventory analysis
        
        Args:
            current_inventory: Current inventory data
            forecast: Forecast data
            
        Returns:
            Analysis results
        """
        inventory_service = self.get_inventory_analyzer()
        if inventory_service:
            return inventory_service.analyze_inventory_levels(current_inventory, forecast)
        else:
            logger.error("InventoryAnalyzerService not available")
            return []
    
    def generate_forecast(self, style_data, horizon_days=None):
        """
        Convenience method for generating forecasts
        
        Args:
            style_data: Historical style data
            horizon_days: Forecast horizon
            
        Returns:
            Forecast results
        """
        forecast_service = self.get_sales_forecaster()
        if forecast_service:
            return forecast_service.forecast_with_consistency(style_data, horizon_days)
        else:
            logger.error("SalesForecastingService not available")
            return {'forecast': 0, 'confidence': 0, 'method': 'unavailable'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all services
        
        Returns:
            Dictionary with service statuses
        """
        status = {
            'initialized': self.initialized,
            'total_services': len(self.services),
            'services': {}
        }
        
        for name, service in self.services.items():
            service_status = {
                'available': True,
                'class': service.__class__.__name__
            }
            
            # Add service-specific status
            if name == 'inventory':
                service_status['config'] = {
                    'safety_stock_multiplier': service.safety_stock_multiplier,
                    'lead_time_days': service.lead_time_days
                }
            elif name == 'forecasting':
                service_status['config'] = {
                    'forecast_horizon': service.forecast_horizon,
                    'target_accuracy': service.target_accuracy,
                    'ml_available': service.ML_AVAILABLE,
                    'engines': list(service.ml_engines.keys())
                }
            
            status['services'][name] = service_status
        
        return status
    
    def perform_integrated_analysis(self, inventory_data, sales_history):
        """
        Perform integrated analysis using multiple services
        
        Args:
            inventory_data: Current inventory data
            sales_history: Historical sales data by style
            
        Returns:
            Integrated analysis results
        """
        results = {
            'timestamp': None,
            'inventory_analysis': None,
            'forecasts': {},
            'recommendations': []
        }
        
        # Generate forecasts for all styles
        forecast_service = self.get_sales_forecaster()
        if forecast_service and sales_history:
            forecast_summary = forecast_service.get_forecast_summary(sales_history)
            results['forecasts'] = forecast_summary
            
            # Create forecast dictionary for inventory analysis
            forecast_dict = {}
            for style_id, forecast_data in forecast_summary.get('forecasts', {}).items():
                forecast_dict[style_id] = forecast_data.get('forecast', 0)
        else:
            forecast_dict = {}
        
        # Analyze inventory with forecasts
        inventory_service = self.get_inventory_analyzer()
        if inventory_service and inventory_data:
            results['inventory_analysis'] = inventory_service.analyze_inventory_levels(
                inventory_data, 
                forecast_dict
            )
            
            # Generate recommendations
            critical_items = inventory_service.get_critical_items(inventory_data, forecast_dict)
            for item in critical_items:
                results['recommendations'].append({
                    'product_id': item['product_id'],
                    'action': 'URGENT REORDER',
                    'quantity': item['reorder_quantity'],
                    'risk': item['shortage_risk']
                })
        
        return results
    
    def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down ServiceManager...")
        
        # Cleanup services if needed
        for name, service in self.services.items():
            logger.info(f"  Shutting down {name} service")
            # Add any cleanup logic here
        
        self.services.clear()
        self.initialized = False
        logger.info("ServiceManager shutdown complete")


# Singleton instance
_instance = None

def get_service_manager(config: Optional[Dict[str, Any]] = None) -> ServiceManager:
    """
    Get singleton instance of ServiceManager
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        ServiceManager instance
    """
    global _instance
    if _instance is None:
        _instance = ServiceManager(config)
    return _instance


def test_service_manager():
    """Test the service manager"""
    print("=" * 80)
    print("Testing ServiceManager")
    print("=" * 80)
    
    # Create service manager with custom config
    config = {
        'inventory': {
            'safety_stock_multiplier': 2.0,
            'lead_time_days': 45
        },
        'forecast': {
            'horizon_days': 60,
            'target_accuracy': 0.90
        }
    }
    
    manager = ServiceManager(config)
    
    # Get system status
    print("\nSystem Status:")
    status = manager.get_system_status()
    print(f"  Initialized: {status['initialized']}")
    print(f"  Total Services: {status['total_services']}")
    
    for name, service_status in status['services'].items():
        print(f"\n  {name.upper()} Service:")
        print(f"    Available: {service_status['available']}")
        print(f"    Class: {service_status['class']}")
        if 'config' in service_status:
            print(f"    Config: {service_status['config']}")
    
    # Test integrated analysis
    print("\n" + "-" * 80)
    print("Testing Integrated Analysis:")
    
    # Sample data
    inventory_data = [
        {'product_id': 'YARN001', 'quantity': 100},
        {'product_id': 'YARN002', 'quantity': 50},
        {'product_id': 'YARN003', 'quantity': 200}
    ]
    
    import pandas as pd
    sales_history = {
        'YARN001': pd.DataFrame({'quantity': [90, 95, 100, 105, 98]}),
        'YARN002': pd.DataFrame({'quantity': [40, 45, 50, 55, 48]}),
        'YARN003': pd.DataFrame({'quantity': [180, 190, 200, 210, 195]})
    }
    
    # Perform integrated analysis
    results = manager.perform_integrated_analysis(inventory_data, sales_history)
    
    print(f"\nForecasts Generated: {len(results['forecasts'].get('forecasts', {}))}")
    if results['forecasts']:
        print(f"  Average Confidence: {results['forecasts'].get('average_confidence', 0):.3f}")
        print(f"  Method Distribution: {results['forecasts'].get('method_distribution', {})}")
    
    if results['inventory_analysis']:
        print(f"\nInventory Analysis: {len(results['inventory_analysis'])} items analyzed")
        critical = sum(1 for item in results['inventory_analysis'] 
                      if item['shortage_risk'] == 'CRITICAL')
        print(f"  Critical Items: {critical}")
    
    if results['recommendations']:
        print(f"\nRecommendations: {len(results['recommendations'])}")
        for rec in results['recommendations'][:3]:
            print(f"  - {rec['product_id']}: {rec['action']} ({rec['quantity']:.0f} units)")
    
    # Test service access
    print("\n" + "-" * 80)
    print("Testing Direct Service Access:")
    
    inventory_service = manager.get_inventory_analyzer()
    if inventory_service:
        print(f"✓ Inventory Analyzer accessible")
        print(f"  Safety Stock Multiplier: {inventory_service.safety_stock_multiplier}")
    
    forecast_service = manager.get_sales_forecaster()
    if forecast_service:
        print(f"✓ Sales Forecaster accessible")
        print(f"  Forecast Horizon: {forecast_service.forecast_horizon} days")
        print(f"  ML Engines: {list(forecast_service.ml_engines.keys())}")
    
    # Shutdown
    manager.shutdown()
    print("\n✓ ServiceManager shutdown complete")
    
    print("\n" + "=" * 80)
    print("✅ ServiceManager test complete")


if __name__ == "__main__":
    test_service_manager()