#!/usr/bin/env python3
"""
Beverly Knits ERP - Optimized Service Manager
Integrates all services with memory optimization and caching
Part of Phase 3: Performance Optimization
"""

import logging
import gc
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimization modules
sys.path.insert(0, str(Path(__file__).parent.parent / "optimization"))
from memory_optimizer import MemoryOptimizer, memory_efficient, limit_dataframe_size

# Import extracted services
from services.inventory_analyzer_service import InventoryAnalyzerService, InventoryConfig
from services.sales_forecasting_service import SalesForecastingService, ForecastConfig
from services.capacity_planning_service import CapacityPlanningService, CapacityConfig
from services.inventory_pipeline_service import InventoryManagementPipelineService, PipelineConfig
from services.yarn_requirement_service import YarnRequirementCalculatorService, YarnRequirementConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedServiceManager:
    """
    Optimized service manager with memory management and caching
    Manages all extracted services with performance optimizations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimized service manager
        
        Args:
            config: Optional configuration dictionary for services
        """
        self.config = config or {}
        self.services = {}
        self.initialized = False
        
        # Initialize memory optimizer
        self.memory_optimizer = MemoryOptimizer()
        self.memory_optimizer.setup_automatic_cleanup(interval_minutes=10)
        
        logger.info("OptimizedServiceManager initializing with memory optimization...")
        
        # Initialize all services with optimization
        self._initialize_services()
        
        # Perform initial garbage collection
        self.memory_optimizer.collect_garbage()
        
        logger.info(f"OptimizedServiceManager ready with {len(self.services)} services")
        self._log_memory_status()
    
    @memory_efficient
    def _initialize_services(self):
        """Initialize all available services with memory optimization"""
        
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
            pipeline_service.set_dependencies(
                inventory_analyzer=self.services.get('inventory'),
                supply_chain_ai=None
            )
            self.services['pipeline'] = pipeline_service
            logger.info("  ✓ InventoryManagementPipelineService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize InventoryManagementPipelineService: {e}")
        
        # Initialize Yarn Requirement Calculator Service
        try:
            yarn_config = YarnRequirementConfig(
                critical_threshold=self.config.get('yarn', {}).get('critical_threshold', 1000),
                high_priority_factor=self.config.get('yarn', {}).get('high_priority_factor', 0.5)
            )
            self.services['yarn'] = YarnRequirementCalculatorService(yarn_config)
            logger.info("  ✓ YarnRequirementCalculatorService initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize YarnRequirementCalculatorService: {e}")
        
        self.initialized = True
    
    def get_service(self, service_name: str):
        """Get a specific service by name"""
        return self.services.get(service_name)
    
    @memory_efficient
    @limit_dataframe_size(max_rows=50000)
    def perform_integrated_analysis(self, 
                                   inventory_data: pd.DataFrame = None,
                                   sales_history: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Perform integrated analysis with memory optimization
        
        Args:
            inventory_data: Current inventory data
            sales_history: Historical sales data
            
        Returns:
            Integrated analysis results with memory optimization
        """
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'memory_before_mb': self.memory_optimizer.get_memory_usage()
        }
        
        # Optimize DataFrames if provided
        if inventory_data is not None:
            inventory_data = self.memory_optimizer.optimize_dataframe(
                inventory_data, "inventory_data"
            )
        
        if sales_history is not None:
            sales_history = self.memory_optimizer.optimize_dataframe(
                sales_history, "sales_history"
            )
        
        # Run pipeline analysis
        pipeline_service = self.services.get('pipeline')
        if pipeline_service:
            try:
                pipeline_results = pipeline_service.run_complete_analysis(
                    sales_data=sales_history,
                    inventory_data=inventory_data
                )
                results['pipeline_analysis'] = pipeline_results
            except Exception as e:
                logger.error(f"Pipeline analysis failed: {e}")
                results['pipeline_error'] = str(e)
        
        # Get yarn requirements
        yarn_service = self.services.get('yarn')
        if yarn_service:
            try:
                yarn_requirements = yarn_service.process_yarn_requirements()
                critical_yarns = yarn_service.get_critical_yarns()
                results['yarn_analysis'] = {
                    'total_yarns': len(yarn_requirements),
                    'critical_yarns': len(critical_yarns),
                    'top_critical': critical_yarns[:5] if critical_yarns else []
                }
            except Exception as e:
                logger.error(f"Yarn analysis failed: {e}")
                results['yarn_error'] = str(e)
        
        # Get capacity analysis
        capacity_service = self.services.get('capacity')
        if capacity_service:
            try:
                metrics = capacity_service.get_capacity_metrics()
                results['capacity_metrics'] = metrics
            except Exception as e:
                logger.error(f"Capacity analysis failed: {e}")
                results['capacity_error'] = str(e)
        
        # Collect garbage after analysis
        gc_stats = self.memory_optimizer.collect_garbage()
        results['memory_after_mb'] = self.memory_optimizer.get_memory_usage()
        results['memory_freed_mb'] = gc_stats.get('memory_freed_mb', 0)
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health including memory status
        
        Returns:
            System health metrics
        """
        health = {
            'initialized': self.initialized,
            'total_services': len(self.services),
            'services': {},
            'memory': self.memory_optimizer.get_memory_report(),
            'performance': {}
        }
        
        # Check each service
        for name, service in self.services.items():
            try:
                health['services'][name] = {
                    'available': True,
                    'class': service.__class__.__name__
                }
            except Exception as e:
                health['services'][name] = {
                    'available': False,
                    'error': str(e)
                }
        
        # Add performance metrics
        health['performance'] = {
            'gc_collections': gc.get_count(),
            'tracked_objects': len(gc.get_objects())
        }
        
        # Generate health score
        health['health_score'] = self._calculate_health_score(health)
        
        return health
    
    def _calculate_health_score(self, health: Dict[str, Any]) -> int:
        """Calculate overall health score (0-100)"""
        score = 100
        
        # Deduct for unavailable services
        unavailable = sum(1 for s in health['services'].values() if not s.get('available'))
        score -= unavailable * 10
        
        # Deduct for high memory usage
        memory_percent = health['memory'].get('percent', 0)
        if memory_percent > 80:
            score -= 20
        elif memory_percent > 60:
            score -= 10
        
        # Deduct for memory growth
        memory_growth = health['memory'].get('memory_growth_mb', 0)
        if memory_growth > 100:
            score -= 15
        elif memory_growth > 50:
            score -= 5
        
        return max(0, score)
    
    def optimize_all_services(self):
        """Run optimization on all services"""
        logger.info("Running optimization on all services...")
        
        # Force garbage collection
        before_memory = self.memory_optimizer.get_memory_usage()
        gc_stats = self.memory_optimizer.collect_garbage(force=True)
        
        # Log optimization results
        logger.info(f"Optimization complete:")
        logger.info(f"  Memory before: {before_memory:.2f} MB")
        logger.info(f"  Memory after: {self.memory_optimizer.get_memory_usage():.2f} MB")
        logger.info(f"  Memory freed: {gc_stats.get('memory_freed_mb', 0):.2f} MB")
        
        return gc_stats
    
    def _log_memory_status(self):
        """Log current memory status"""
        stats = self.memory_optimizer.get_memory_stats()
        logger.info(f"Memory Status: {stats['rss_mb']:.2f} MB ({stats['percent']:.1f}%), "
                   f"Objects: {stats['tracked_objects']}")
    
    def shutdown(self):
        """Gracefully shutdown all services with cleanup"""
        logger.info("Shutting down OptimizedServiceManager...")
        
        # Cleanup services
        for name, service in self.services.items():
            logger.info(f"  Shutting down {name} service")
        
        self.services.clear()
        
        # Final memory cleanup
        self.memory_optimizer.shutdown()
        
        self.initialized = False
        logger.info("OptimizedServiceManager shutdown complete")


def test_optimized_service_manager():
    """Test the optimized service manager"""
    print("=" * 80)
    print("Testing OptimizedServiceManager")
    print("=" * 80)
    
    # Create manager with custom config
    config = {
        'inventory': {'safety_stock_multiplier': 1.5, 'lead_time_days': 30},
        'forecast': {'horizon_days': 60, 'target_accuracy': 0.90},
        'capacity': {'bottleneck_threshold': 0.85, 'critical_threshold': 0.95},
        'pipeline': {'waste_factor': 1.2, 'growth_forecast': 1.1},
        'yarn': {'critical_threshold': 500}
    }
    
    manager = OptimizedServiceManager(config)
    
    # Test 1: System health
    print("\n1. System Health Check:")
    health = manager.get_system_health()
    print(f"  Health Score: {health['health_score']}/100")
    print(f"  Services Available: {health['total_services']}")
    print(f"  Memory Usage: {health['memory']['rss_mb']:.2f} MB")
    print(f"  Memory Growth: {health['memory']['memory_growth_mb']:.2f} MB")
    
    # Test 2: Service availability
    print("\n2. Service Status:")
    for name, status in health['services'].items():
        symbol = "✓" if status['available'] else "✗"
        print(f"  {symbol} {name}: {status['class'] if status['available'] else status.get('error', 'Unknown')}")
    
    # Test 3: Integrated analysis with sample data
    print("\n3. Integrated Analysis:")
    
    # Create sample data
    inventory_data = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'] * 100,
        'Planning Balance': [100, 50, 200] * 100,
        'Stock': [80, 40, 180] * 100
    })
    
    sales_history = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'] * 100,
        'Consumed': [-90, -45, -180] * 100,
        'Sales': [100, 50, 200] * 100
    })
    
    results = manager.perform_integrated_analysis(inventory_data, sales_history)
    
    print(f"  Memory Before: {results['memory_before_mb']:.2f} MB")
    print(f"  Memory After: {results['memory_after_mb']:.2f} MB")
    print(f"  Memory Freed: {results.get('memory_freed_mb', 0):.2f} MB")
    
    if 'pipeline_analysis' in results:
        pipeline = results['pipeline_analysis']
        if 'summary' in pipeline:
            print(f"  Pipeline Analysis: {pipeline['summary'].get('components_analyzed', 0)} components")
    
    if 'yarn_analysis' in results:
        yarn = results['yarn_analysis']
        print(f"  Yarn Analysis: {yarn['total_yarns']} yarns, {yarn['critical_yarns']} critical")
    
    # Test 4: Optimization
    print("\n4. Running Optimization:")
    opt_stats = manager.optimize_all_services()
    print(f"  Objects Collected: {opt_stats.get('objects_collected', 0)}")
    print(f"  Memory Freed: {opt_stats.get('memory_freed_mb', 0):.2f} MB")
    
    # Test 5: Memory recommendations
    print("\n5. Memory Recommendations:")
    memory_report = manager.get_system_health()['memory']
    if memory_report.get('recommendations'):
        for rec in memory_report['recommendations']:
            print(f"  - {rec}")
    else:
        print("  No recommendations (memory usage optimal)")
    
    # Shutdown
    manager.shutdown()
    print("\n✓ OptimizedServiceManager test complete")
    
    print("\n" + "=" * 80)
    print("✅ All tests complete")


if __name__ == "__main__":
    test_optimized_service_manager()