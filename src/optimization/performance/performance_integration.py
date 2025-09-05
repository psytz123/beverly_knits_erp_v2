"""
Performance Integration - Integrate all optimizers into existing services
Provides seamless integration with the monolith and new services
"""
import pandas as pd
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import time

from .dataframe_optimizer import DataFrameOptimizer
from .query_optimizer import QueryOptimizer
from .async_processor import AsyncProcessor, BackgroundScheduler
from .memory_optimizer import MemoryOptimizer, ConnectionPoolOptimizer

logger = logging.getLogger(__name__)


class PerformanceIntegration:
    """Central integration point for all performance optimizations"""
    
    def __init__(self):
        self.df_optimizer = DataFrameOptimizer()
        self.query_optimizer = QueryOptimizer()
        self.async_processor = AsyncProcessor()
        self.memory_optimizer = MemoryOptimizer()
        self.connection_pool = ConnectionPoolOptimizer()
        self.background_scheduler = BackgroundScheduler()
        self.optimization_stats = {
            'dataframe_optimizations': 0,
            'query_optimizations': 0,
            'async_conversions': 0,
            'memory_optimizations': 0,
            'total_time_saved_ms': 0
        }
    
    def optimize_inventory_service(self, inventory_service: Any) -> None:
        """
        Optimize InventoryAnalyzer and related services
        Replaces iterrows with vectorized operations
        """
        original_methods = {}
        
        # Replace calculate_planning_balance
        if hasattr(inventory_service, 'calculate_planning_balance'):
            original_methods['calculate_planning_balance'] = inventory_service.calculate_planning_balance
            
            def optimized_planning_balance(df):
                start = time.perf_counter()
                
                # Use vectorized optimization
                result = DataFrameOptimizer.optimize_planning_balance_calculation(df)
                result = MemoryOptimizer.optimize_dataframe(result)
                
                elapsed = (time.perf_counter() - start) * 1000
                self.optimization_stats['dataframe_optimizations'] += 1
                self.optimization_stats['total_time_saved_ms'] += max(0, 100 - elapsed)  # Estimate
                
                logger.info(f"Optimized planning balance calculation in {elapsed:.1f}ms")
                return result
            
            inventory_service.calculate_planning_balance = optimized_planning_balance
        
        # Replace detect_shortages
        if hasattr(inventory_service, 'detect_shortages'):
            original_methods['detect_shortages'] = inventory_service.detect_shortages
            
            def optimized_shortage_detection(df, threshold='min_stock'):
                # Use vectorized shortage detection
                result = DataFrameOptimizer.optimize_shortage_detection(df, threshold)
                self.optimization_stats['dataframe_optimizations'] += 1
                return result
            
            inventory_service.detect_shortages = optimized_shortage_detection
        
        # Store original methods for rollback
        inventory_service._original_methods = original_methods
        logger.info(f"Optimized {len(original_methods)} methods in inventory service")
    
    def optimize_production_service(self, production_service: Any) -> None:
        """
        Optimize production planning and scheduling
        """
        # Replace BOM explosion calculations
        if hasattr(production_service, 'explode_bom'):
            original_explode = production_service.explode_bom
            
            def optimized_bom_explosion(bom_df, quantity, style_id=None):
                result = DataFrameOptimizer.optimize_bom_explosion(bom_df, quantity, style_id)
                self.optimization_stats['dataframe_optimizations'] += 1
                return result
            
            production_service.explode_bom = optimized_bom_explosion
        
        # Replace production scheduling
        if hasattr(production_service, 'schedule_orders'):
            original_schedule = production_service.schedule_orders
            
            def optimized_scheduling(orders_df, capacity_df):
                result = DataFrameOptimizer.optimize_production_scheduling(orders_df, capacity_df)
                self.optimization_stats['dataframe_optimizations'] += 1
                return result
            
            production_service.schedule_orders = optimized_scheduling
        
        logger.info("Optimized production service methods")
    
    def optimize_yarn_service(self, yarn_service: Any) -> None:
        """
        Optimize yarn allocation and management
        """
        if hasattr(yarn_service, 'allocate_yarn'):
            original_allocate = yarn_service.allocate_yarn
            
            def optimized_allocation(yarn_df, orders_df):
                result = DataFrameOptimizer.optimize_yarn_allocation(yarn_df, orders_df)
                result = MemoryOptimizer.optimize_dataframe(result)
                self.optimization_stats['dataframe_optimizations'] += 1
                return result
            
            yarn_service.allocate_yarn = optimized_allocation
        
        logger.info("Optimized yarn service methods")
    
    def optimize_data_queries(self, data_loader: Any) -> None:
        """
        Optimize database queries in data loaders
        """
        if hasattr(data_loader, 'load_yarn_inventory'):
            original_load = data_loader.load_yarn_inventory
            
            def optimized_load_yarn():
                # Use optimized query
                query, params = self.query_optimizer.optimize_yarn_query()
                
                # Execute with connection pool
                with self.connection_pool.get_connection() as conn:
                    df = pd.read_sql(query, conn, params=params)
                
                # Optimize memory
                df = MemoryOptimizer.optimize_dataframe(df)
                
                self.optimization_stats['query_optimizations'] += 1
                self.optimization_stats['memory_optimizations'] += 1
                return df
            
            data_loader.load_yarn_inventory = optimized_load_yarn
        
        logger.info("Optimized data loader queries")
    
    async def convert_to_async(self, service: Any) -> None:
        """
        Convert blocking operations to async in a service
        """
        methods_converted = 0
        
        for method_name in dir(service):
            if not method_name.startswith('_'):
                method = getattr(service, method_name)
                
                if callable(method) and not asyncio.iscoroutinefunction(method):
                    # Check if method has blocking operations
                    if self._has_blocking_operations(method):
                        # Create async version
                        async_method = self._make_async_method(method)
                        setattr(service, f"{method_name}_async", async_method)
                        methods_converted += 1
        
        self.optimization_stats['async_conversions'] += methods_converted
        logger.info(f"Converted {methods_converted} methods to async in {service.__class__.__name__}")
    
    def _has_blocking_operations(self, method: Callable) -> bool:
        """Check if method has blocking operations"""
        try:
            code_str = str(method.__code__.co_code)
            blocking_patterns = ['time.sleep', 'requests.', 'urllib.', '.read()', '.write()']
            return any(pattern in code_str for pattern in blocking_patterns)
        except:
            return False
    
    def _make_async_method(self, method: Callable) -> Callable:
        """Convert method to async"""
        @wraps(method)
        async def async_method(*args, **kwargs):
            return await self.async_processor.run_io_operation(method, *args, **kwargs)
        return async_method
    
    def apply_all_optimizations(self, app: Any) -> Dict[str, Any]:
        """
        Apply all optimizations to the Flask app
        Returns optimization report
        """
        start_time = time.perf_counter()
        
        # Create database indexes
        created_indexes = self.query_optimizer.create_missing_indexes()
        
        # Optimize services if they exist
        if hasattr(app, 'inventory_service'):
            self.optimize_inventory_service(app.inventory_service)
        
        if hasattr(app, 'production_service'):
            self.optimize_production_service(app.production_service)
        
        if hasattr(app, 'yarn_service'):
            self.optimize_yarn_service(app.yarn_service)
        
        if hasattr(app, 'data_loader'):
            self.optimize_data_queries(app.data_loader)
        
        # Set up background tasks
        self._setup_background_tasks()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return {
            'status': 'success',
            'optimizations_applied': self.optimization_stats,
            'indexes_created': len(created_indexes),
            'integration_time_ms': elapsed,
            'memory_report': self.memory_optimizer.get_memory_report()
        }
    
    def _setup_background_tasks(self):
        """Set up background optimization tasks"""
        # Memory cleanup task
        self.background_scheduler.add_periodic_task(
            'memory_cleanup',
            lambda: self.memory_optimizer.cleanup_memory(),
            interval=3600  # Every hour
        )
        
        # Query cache cleanup
        self.background_scheduler.add_periodic_task(
            'query_cache_cleanup',
            lambda: self.query_optimizer.clear_cache(),
            interval=7200  # Every 2 hours
        )
        
        # Connection pool cleanup
        self.background_scheduler.add_periodic_task(
            'connection_pool_cleanup',
            lambda: self.connection_pool.cleanup_idle_connections(),
            interval=1800  # Every 30 minutes
        )
        
        asyncio.create_task(self.background_scheduler.start())
        logger.info("Background optimization tasks scheduled")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            'optimization_stats': self.optimization_stats,
            'memory_report': self.memory_optimizer.get_memory_report(),
            'query_stats': self.query_optimizer.analyze_slow_queries(),
            'connection_pool': self.connection_pool.get_pool_stats(),
            'background_tasks': [
                self.background_scheduler.get_task_status(name)
                for name in ['memory_cleanup', 'query_cache_cleanup', 'connection_pool_cleanup']
            ]
        }


class OptimizationMiddleware:
    """Flask middleware to apply optimizations automatically"""
    
    def __init__(self, app, integration: PerformanceIntegration):
        self.app = app
        self.integration = integration
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up Flask hooks for optimization"""
        
        @self.app.before_request
        def before_request():
            """Prepare optimizations before request"""
            # Clear old cache entries
            if hasattr(g, 'request_count'):
                g.request_count += 1
                if g.request_count % 100 == 0:
                    self.integration.query_optimizer.clear_cache()
            else:
                g.request_count = 1
        
        @self.app.after_request
        def after_request(response):
            """Clean up after request"""
            # Add performance headers
            if hasattr(g, 'query_time'):
                response.headers['X-Query-Time'] = str(g.query_time)
            if hasattr(g, 'optimization_applied'):
                response.headers['X-Optimization'] = 'true'
            return response
        
        @self.app.teardown_appcontext
        def teardown(error):
            """Clean up resources"""
            # Force garbage collection for large responses
            import gc
            gc.collect()


def apply_performance_optimizations(app):
    """
    Main entry point to apply all performance optimizations
    
    Usage:
        from src.optimization.performance.performance_integration import apply_performance_optimizations
        
        app = Flask(__name__)
        # ... app setup ...
        
        # Apply optimizations
        report = apply_performance_optimizations(app)
        print(f"Optimizations applied: {report}")
    """
    integration = PerformanceIntegration()
    
    # Apply optimizations
    report = integration.apply_all_optimizations(app)
    
    # Add middleware
    OptimizationMiddleware(app, integration)
    
    # Log results
    logger.info(f"Performance optimizations applied successfully")
    logger.info(f"Report: {report}")
    
    return report


def optimize_dataframe_operation(func: Callable) -> Callable:
    """
    Decorator to automatically optimize DataFrame operations
    
    Usage:
        @optimize_dataframe_operation
        def process_inventory(df):
            # Your processing code
            return df
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find DataFrames in arguments
        optimized_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Optimize before processing
                arg = MemoryOptimizer.optimize_dataframe(arg)
                arg = DataFrameOptimizer.apply_all_optimizations(arg)
            optimized_args.append(arg)
        
        # Execute function
        result = func(*optimized_args, **kwargs)
        
        # Optimize result if it's a DataFrame
        if isinstance(result, pd.DataFrame):
            result = MemoryOptimizer.optimize_dataframe(result)
        
        return result
    
    return wrapper


def benchmark_optimization(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark function with and without optimization
    
    Returns comparison metrics
    """
    import copy
    
    # Run without optimization
    start = time.perf_counter()
    original_result = func(*args, **kwargs)
    original_time = (time.perf_counter() - start) * 1000
    
    # Run with optimization
    optimized_func = optimize_dataframe_operation(func)
    start = time.perf_counter()
    optimized_result = optimized_func(*args, **kwargs)
    optimized_time = (time.perf_counter() - start) * 1000
    
    # Calculate improvements
    improvement_pct = ((original_time - optimized_time) / original_time) * 100 if original_time > 0 else 0
    
    # Memory comparison
    original_memory = 0
    optimized_memory = 0
    
    if isinstance(original_result, pd.DataFrame):
        original_memory = original_result.memory_usage(deep=True).sum() / 1024**2
    if isinstance(optimized_result, pd.DataFrame):
        optimized_memory = optimized_result.memory_usage(deep=True).sum() / 1024**2
    
    memory_improvement = ((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0
    
    return {
        'original_time_ms': original_time,
        'optimized_time_ms': optimized_time,
        'speed_improvement_pct': improvement_pct,
        'original_memory_mb': original_memory,
        'optimized_memory_mb': optimized_memory,
        'memory_improvement_pct': memory_improvement,
        'results_match': str(original_result) == str(optimized_result) if original_result is not None else True
    }