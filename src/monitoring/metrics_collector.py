"""
Metrics Collector
Prometheus metrics collection for system monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CollectorRegistry
from flask import Response, request
import time
import psutil
import logging
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Define metrics
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

api_latency = Histogram(
    'api_latency_seconds',
    'API latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    registry=registry
)

# Business metrics
yarn_shortage_gauge = Gauge(
    'yarn_shortages',
    'Current number of yarn shortages',
    ['severity'],
    registry=registry
)

production_orders_gauge = Gauge(
    'production_orders',
    'Production orders by status',
    ['status'],
    registry=registry
)

inventory_value_gauge = Gauge(
    'inventory_value',
    'Total inventory value in USD',
    registry=registry
)

capacity_utilization_gauge = Gauge(
    'capacity_utilization_percent',
    'Production capacity utilization percentage',
    ['work_center'],
    registry=registry
)

forecast_accuracy_gauge = Gauge(
    'forecast_accuracy_percent',
    'ML forecast accuracy percentage',
    ['model'],
    registry=registry
)

# Cache metrics
cache_operations = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'tier'],
    registry=registry
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['tier'],
    registry=registry
)

# System metrics
system_cpu_percent = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_percent = Gauge(
    'system_memory_percent',
    'System memory usage percentage',
    registry=registry
)

system_disk_percent = Gauge(
    'system_disk_percent',
    'System disk usage percentage',
    registry=registry
)

# Error metrics
error_rate = Counter(
    'errors_total',
    'Total errors',
    ['type', 'severity'],
    registry=registry
)

# Database metrics
db_connections = Gauge(
    'database_connections',
    'Number of database connections',
    ['state'],
    registry=registry
)

db_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)


class MetricsCollector:
    """Collect and manage application metrics"""
    
    def __init__(self, app=None):
        """Initialize metrics collector"""
        self.app = app
        self.start_time = time.time()
        self.request_count = 0
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize metrics with Flask app"""
        self.app = app
        
        # Add before/after request handlers
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Add metrics endpoint
        @app.route('/metrics')
        def metrics():
            """Prometheus metrics endpoint"""
            # Update business metrics
            self.update_business_metrics()
            
            # Update system metrics
            self.update_system_metrics()
            
            # Generate metrics
            return Response(generate_latest(registry), mimetype='text/plain')
        
        # Add health endpoint
        @app.route('/health')
        def health():
            """Health check endpoint"""
            return {
                'status': 'healthy',
                'uptime': time.time() - self.start_time,
                'requests': self.request_count
            }
        
        # Add ready endpoint
        @app.route('/ready')
        def ready():
            """Readiness check endpoint"""
            # Check critical dependencies
            ready_status = self.check_readiness()
            
            if ready_status['ready']:
                return ready_status, 200
            else:
                return ready_status, 503
        
        logger.info("Metrics collector initialized")
    
    def before_request(self):
        """Handler before each request"""
        request.start_time = time.time()
        active_connections.inc()
        self.request_count += 1
    
    def after_request(self, response):
        """Handler after each request"""
        # Calculate request duration
        if hasattr(request, 'start_time'):
            latency = time.time() - request.start_time
            
            # Record metrics
            endpoint = request.endpoint or 'unknown'
            method = request.method
            
            api_latency.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)
            
            api_requests.labels(
                method=method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            
            # Track errors
            if response.status_code >= 400:
                severity = 'error' if response.status_code >= 500 else 'warning'
                error_rate.labels(
                    type='http',
                    severity=severity
                ).inc()
        
        active_connections.dec()
        
        return response
    
    def update_business_metrics(self):
        """Update business-specific metrics"""
        try:
            # Get services
            from src.services.service_container import services
            
            # Update yarn shortage metrics
            try:
                inventory_service = services.get('inventory')
                shortages = inventory_service.detect_shortages()
                
                if hasattr(shortages, '__len__'):
                    total_shortages = len(shortages)
                    
                    # Categorize by severity
                    critical = len([s for s in shortages if s.get('planning_balance', 0) < -1000])
                    warning = total_shortages - critical
                    
                    yarn_shortage_gauge.labels(severity='critical').set(critical)
                    yarn_shortage_gauge.labels(severity='warning').set(warning)
            except Exception as e:
                logger.error(f"Error updating yarn metrics: {e}")
            
            # Update production order metrics
            try:
                data_loader = services.get('data_loader')
                orders = data_loader.load_production_orders()
                
                if orders is not None and not orders.empty:
                    status_counts = orders['status'].value_counts().to_dict()
                    
                    for status, count in status_counts.items():
                        production_orders_gauge.labels(status=status).set(count)
            except Exception as e:
                logger.error(f"Error updating production metrics: {e}")
            
            # Update capacity metrics
            try:
                capacity_service = services.get('capacity')
                capacity_data = capacity_service.get_capacity_analysis()
                
                if 'work_centers' in capacity_data:
                    for wc, util in capacity_data['work_centers'].items():
                        capacity_utilization_gauge.labels(work_center=wc).set(util.get('utilization', 0))
            except Exception as e:
                logger.error(f"Error updating capacity metrics: {e}")
            
            # Update cache metrics
            try:
                cache = services.get('cache')
                cache_stats = cache.get_stats()
                
                if 'hit_rate' in cache_stats:
                    cache_hit_ratio.labels(tier='overall').set(cache_stats['hit_rate'])
                if 'l1_hit_rate' in cache_stats:
                    cache_hit_ratio.labels(tier='l1').set(cache_stats['l1_hit_rate'])
                if 'l2_hit_rate' in cache_stats:
                    cache_hit_ratio.labels(tier='l2').set(cache_stats['l2_hit_rate'])
            except Exception as e:
                logger.error(f"Error updating cache metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error updating business metrics: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_percent.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_percent.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            system_disk_percent.set(disk.percent)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def check_readiness(self) -> Dict[str, Any]:
        """Check if application is ready"""
        ready_status = {
            'ready': True,
            'checks': {}
        }
        
        # Check database
        try:
            from src.services.service_container import services
            data_loader = services.get('data_loader')
            
            # Try to load some data
            data_loader.load_yarn_inventory()
            ready_status['checks']['database'] = 'ok'
        except Exception as e:
            ready_status['checks']['database'] = f'failed: {e}'
            ready_status['ready'] = False
        
        # Check cache
        try:
            cache = services.get('cache')
            cache.get_stats()
            ready_status['checks']['cache'] = 'ok'
        except Exception as e:
            ready_status['checks']['cache'] = f'failed: {e}'
            # Cache failure doesn't prevent readiness
        
        return ready_status
    
    def record_error(self, error_type: str, severity: str = 'error'):
        """Record an error in metrics"""
        error_rate.labels(type=error_type, severity=severity).inc()
    
    def record_db_query(self, query_type: str, duration: float):
        """Record database query metrics"""
        db_query_duration.labels(query_type=query_type).observe(duration)
    
    def record_cache_operation(self, operation: str, tier: str):
        """Record cache operation metrics"""
        cache_operations.labels(operation=operation, tier=tier).inc()


def setup_metrics(app):
    """Setup metrics collection for Flask app"""
    collector = MetricsCollector(app)
    return collector


def monitor_function(name: Optional[str] = None):
    """
    Decorator to monitor function execution
    
    Args:
        name: Optional metric name (uses function name if not provided)
    """
    def decorator(func):
        metric_name = name or func.__name__
        
        # Create function-specific metrics
        function_calls = Counter(
            f'{metric_name}_calls_total',
            f'Total calls to {metric_name}',
            registry=registry
        )
        
        function_duration = Histogram(
            f'{metric_name}_duration_seconds',
            f'Duration of {metric_name} in seconds',
            registry=registry
        )
        
        function_errors = Counter(
            f'{metric_name}_errors_total',
            f'Total errors in {metric_name}',
            registry=registry
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_calls.inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                function_duration.observe(duration)
                return result
                
            except Exception as e:
                function_errors.inc()
                raise
        
        return wrapper
    
    return decorator


class BusinessMetrics:
    """Helper class for business metrics"""
    
    @staticmethod
    def update_inventory_metrics(total_value: float, shortage_count: int):
        """Update inventory-related metrics"""
        inventory_value_gauge.set(total_value)
        yarn_shortage_gauge.labels(severity='total').set(shortage_count)
    
    @staticmethod
    def update_production_metrics(orders_by_status: Dict[str, int]):
        """Update production-related metrics"""
        for status, count in orders_by_status.items():
            production_orders_gauge.labels(status=status).set(count)
    
    @staticmethod
    def update_forecast_metrics(accuracy_by_model: Dict[str, float]):
        """Update forecast accuracy metrics"""
        for model, accuracy in accuracy_by_model.items():
            forecast_accuracy_gauge.labels(model=model).set(accuracy * 100)
    
    @staticmethod
    def update_capacity_metrics(utilization_by_center: Dict[str, float]):
        """Update capacity utilization metrics"""
        for center, utilization in utilization_by_center.items():
            capacity_utilization_gauge.labels(work_center=center).set(utilization)