"""
API Monitoring and Observability
Tracks API performance, errors, and provides dashboards
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    API_CALL = "api_call"
    API_ERROR = "api_error"
    API_LATENCY = "api_latency"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"
    FALLBACK_TRIGGERED = "fallback_triggered"
    DATA_LOADED = "data_loaded"


@dataclass
class Metric:
    """Single metric data point"""
    timestamp: datetime
    type: MetricType
    endpoint: Optional[str]
    value: float
    tags: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'endpoint': self.endpoint,
            'value': self.value,
            'tags': self.tags
        }


class APIMonitor:
    """
    Comprehensive API monitoring and observability
    Tracks performance, errors, and system health
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize monitor
        
        Args:
            max_history: Maximum metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.aggregated_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_success': None,
            'last_error': None
        })
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Performance tracking
        self.response_times = defaultdict(list)
        self.slow_queries = []
        self.slow_threshold = 5.0  # seconds
        
        # Alert configuration
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'response_time': 5.0,  # 5 seconds
            'circuit_breaker_opens': 3,  # 3 opens per hour
            'auth_failures': 5  # 5 auth failures per hour
        }
        
        # Monitoring state
        self.monitoring_enabled = True
        self.start_time = datetime.now()
    
    def record_api_call(
        self,
        endpoint: str,
        duration: float,
        success: bool,
        status_code: Optional[int] = None,
        error: Optional[str] = None
    ):
        """
        Record an API call
        
        Args:
            endpoint: API endpoint called
            duration: Call duration in seconds
            success: Whether call succeeded
            status_code: HTTP status code
            error: Error message if failed
        """
        if not self.monitoring_enabled:
            return
        
        # Create metric
        metric = Metric(
            timestamp=datetime.now(),
            type=MetricType.API_CALL if success else MetricType.API_ERROR,
            endpoint=endpoint,
            value=duration,
            tags={
                'success': success,
                'status_code': status_code,
                'error': error
            }
        )
        
        self.metrics.append(metric)
        
        # Update aggregated stats
        stats = self.aggregated_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += duration
        
        if success:
            stats['last_success'] = datetime.now()
            self.counters['api_calls_success'] += 1
        else:
            stats['errors'] += 1
            stats['last_error'] = datetime.now()
            self.counters['api_calls_failed'] += 1
            self.error_counts[endpoint] += 1
        
        # Track response times
        self.response_times[endpoint].append(duration)
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
        
        # Check for slow queries
        if duration > self.slow_threshold:
            self.slow_queries.append({
                'endpoint': endpoint,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            if len(self.slow_queries) > 10:
                self.slow_queries = self.slow_queries[-10:]
        
        # Check alert conditions
        self._check_alerts(endpoint, duration, success)
        
        logger.debug(f"API call recorded: {endpoint} - {duration:.2f}s - {'Success' if success else 'Failed'}")
    
    def record_cache_access(self, endpoint: str, hit: bool):
        """
        Record cache hit/miss
        
        Args:
            endpoint: Endpoint accessed
            hit: Whether it was a cache hit
        """
        metric = Metric(
            timestamp=datetime.now(),
            type=MetricType.CACHE_HIT if hit else MetricType.CACHE_MISS,
            endpoint=endpoint,
            value=1.0,
            tags={'hit': hit}
        )
        
        self.metrics.append(metric)
        
        if hit:
            self.counters['cache_hits'] += 1
        else:
            self.counters['cache_misses'] += 1
    
    def record_auth_event(self, success: bool, username: Optional[str] = None):
        """
        Record authentication event
        
        Args:
            success: Whether authentication succeeded
            username: Username attempted
        """
        metric = Metric(
            timestamp=datetime.now(),
            type=MetricType.AUTH_SUCCESS if success else MetricType.AUTH_FAILURE,
            endpoint='auth',
            value=1.0,
            tags={'username': username}
        )
        
        self.metrics.append(metric)
        
        if success:
            self.counters['auth_success'] += 1
        else:
            self.counters['auth_failures'] += 1
            
            # Check for excessive auth failures
            recent_failures = self._count_recent_metrics(MetricType.AUTH_FAILURE, hours=1)
            if recent_failures > self.alert_thresholds['auth_failures']:
                self._send_alert('auth_failures', f"High auth failure rate: {recent_failures} in last hour")
    
    def record_circuit_breaker_event(self, opened: bool, endpoint: str):
        """
        Record circuit breaker state change
        
        Args:
            opened: Whether circuit was opened (True) or closed (False)
            endpoint: Affected endpoint
        """
        metric = Metric(
            timestamp=datetime.now(),
            type=MetricType.CIRCUIT_OPEN if opened else MetricType.CIRCUIT_CLOSE,
            endpoint=endpoint,
            value=1.0,
            tags={'state': 'open' if opened else 'closed'}
        )
        
        self.metrics.append(metric)
        
        if opened:
            self.counters['circuit_breaker_opens'] += 1
            
            # Check for excessive circuit breaker activations
            recent_opens = self._count_recent_metrics(MetricType.CIRCUIT_OPEN, hours=1)
            if recent_opens > self.alert_thresholds['circuit_breaker_opens']:
                self._send_alert('circuit_breaker', f"Circuit breaker opened {recent_opens} times in last hour")
    
    def record_fallback(self, data_type: str, reason: str):
        """
        Record fallback to file loading
        
        Args:
            data_type: Type of data being loaded
            reason: Reason for fallback
        """
        metric = Metric(
            timestamp=datetime.now(),
            type=MetricType.FALLBACK_TRIGGERED,
            endpoint=data_type,
            value=1.0,
            tags={'reason': reason}
        )
        
        self.metrics.append(metric)
        self.counters['fallback_triggered'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics
        
        Returns:
            Dictionary of statistics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate cache hit rate
        cache_total = self.counters['cache_hits'] + self.counters['cache_misses']
        cache_hit_rate = (self.counters['cache_hits'] / cache_total * 100) if cache_total > 0 else 0
        
        # Calculate API success rate
        api_total = self.counters['api_calls_success'] + self.counters['api_calls_failed']
        api_success_rate = (self.counters['api_calls_success'] / api_total * 100) if api_total > 0 else 0
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)
        
        return {
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat(),
            'total_api_calls': api_total,
            'api_success_rate': api_success_rate,
            'cache_hit_rate': cache_hit_rate,
            'total_errors': self.counters['api_calls_failed'],
            'auth_failures': self.counters['auth_failures'],
            'circuit_breaker_opens': self.counters['circuit_breaker_opens'],
            'fallback_triggered': self.counters['fallback_triggered'],
            'average_response_times': avg_response_times,
            'slow_queries': self.slow_queries,
            'endpoint_stats': self._get_endpoint_stats()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status
        
        Returns:
            Health status dictionary
        """
        stats = self.get_statistics()
        
        # Determine health score
        health_score = 100
        issues = []
        
        # Check API success rate
        if stats['api_success_rate'] < 95:
            health_score -= 20
            issues.append(f"Low API success rate: {stats['api_success_rate']:.1f}%")
        
        # Check cache hit rate
        if stats['cache_hit_rate'] < 50:
            health_score -= 10
            issues.append(f"Low cache hit rate: {stats['cache_hit_rate']:.1f}%")
        
        # Check for recent errors
        recent_errors = self._count_recent_metrics(MetricType.API_ERROR, minutes=5)
        if recent_errors > 10:
            health_score -= 15
            issues.append(f"High error rate: {recent_errors} errors in last 5 minutes")
        
        # Check for circuit breaker issues
        if stats['circuit_breaker_opens'] > 0:
            health_score -= 10
            issues.append(f"Circuit breaker activated {stats['circuit_breaker_opens']} times")
        
        # Determine status
        if health_score >= 90:
            status = 'healthy'
        elif health_score >= 70:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'health_score': health_score,
            'issues': issues,
            'metrics': {
                'api_success_rate': stats['api_success_rate'],
                'cache_hit_rate': stats['cache_hit_rate'],
                'recent_errors': recent_errors,
                'uptime': stats['uptime_seconds']
            }
        }
    
    def get_recent_metrics(self, minutes: int = 5) -> List[Dict]:
        """
        Get recent metrics
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            List of recent metrics
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [m for m in self.metrics if m.timestamp > cutoff]
        return [m.to_dict() for m in recent]
    
    def _get_endpoint_stats(self) -> Dict[str, Dict]:
        """Get per-endpoint statistics"""
        endpoint_stats = {}
        
        for endpoint, stats in self.aggregated_stats.items():
            if stats['count'] > 0:
                endpoint_stats[endpoint] = {
                    'total_calls': stats['count'],
                    'error_count': stats['errors'],
                    'error_rate': (stats['errors'] / stats['count'] * 100),
                    'avg_response_time': stats['total_time'] / stats['count'],
                    'last_success': stats['last_success'].isoformat() if stats['last_success'] else None,
                    'last_error': stats['last_error'].isoformat() if stats['last_error'] else None
                }
        
        return endpoint_stats
    
    def _count_recent_metrics(
        self,
        metric_type: MetricType,
        minutes: int = 0,
        hours: int = 0
    ) -> int:
        """Count recent metrics of a specific type"""
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
        else:
            cutoff = datetime.now() - timedelta(minutes=minutes)
        
        count = sum(1 for m in self.metrics 
                   if m.type == metric_type and m.timestamp > cutoff)
        return count
    
    def _check_alerts(self, endpoint: str, duration: float, success: bool):
        """Check if any alert conditions are met"""
        # Check response time
        if duration > self.alert_thresholds['response_time']:
            self._send_alert('slow_response', f"Slow API response: {endpoint} took {duration:.2f}s")
        
        # Check error rate for endpoint
        if endpoint in self.aggregated_stats:
            stats = self.aggregated_stats[endpoint]
            if stats['count'] >= 10:  # Only check after enough samples
                error_rate = stats['errors'] / stats['count']
                if error_rate > self.alert_thresholds['error_rate']:
                    self._send_alert('high_error_rate', 
                                    f"High error rate for {endpoint}: {error_rate:.1%}")
    
    def _send_alert(self, alert_type: str, message: str):
        """
        Send an alert
        
        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message
        }
        
        # Log the alert
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # TODO: Implement webhook/email/SMS alerting
        # For now, just log to file
        try:
            alerts_file = Path('logs/api_alerts.json')
            alerts_file.parent.mkdir(exist_ok=True)
            
            # Load existing alerts
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            # Add new alert
            alerts.append(alert)
            
            # Keep only recent alerts
            if len(alerts) > 100:
                alerts = alerts[-100:]
            
            # Save alerts
            with open(alerts_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to file
        
        Args:
            filepath: Path to export file
        """
        try:
            metrics_data = {
                'export_time': datetime.now().isoformat(),
                'statistics': self.get_statistics(),
                'health': self.get_health_status(),
                'recent_metrics': self.get_recent_metrics(60),
                'alerts': []
            }
            
            # Load alerts if they exist
            alerts_file = Path('logs/api_alerts.json')
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    metrics_data['alerts'] = json.load(f)
            
            # Save metrics
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.aggregated_stats.clear()
        self.counters.clear()
        self.error_counts.clear()
        self.response_times.clear()
        self.slow_queries.clear()
        self.start_time = datetime.now()
        logger.info("Metrics reset")


# Global monitor instance
_monitor = None


def get_monitor() -> APIMonitor:
    """Get or create global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = APIMonitor()
    return _monitor


def monitor_api_call(endpoint: str):
    """
    Decorator to monitor API calls
    
    Usage:
        @monitor_api_call('/api/yarn/active')
        async def get_yarn_data():
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = False
            error = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                monitor.record_api_call(endpoint, duration, success, error=error)
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = False
            error = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                monitor.record_api_call(endpoint, duration, success, error=error)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # Test monitoring
    monitor = get_monitor()
    
    # Simulate some API calls
    monitor.record_api_call('/api/yarn/active', 0.5, True, 200)
    monitor.record_api_call('/api/yarn/active', 0.3, True, 200)
    monitor.record_api_call('/api/knit-orders', 1.2, True, 200)
    monitor.record_api_call('/api/knit-orders', 0.0, False, 500, "Server error")
    
    # Simulate cache access
    monitor.record_cache_access('/api/yarn/active', True)
    monitor.record_cache_access('/api/knit-orders', False)
    
    # Simulate auth events
    monitor.record_auth_event(True, 'test_user')
    monitor.record_auth_event(False, 'bad_user')
    
    # Get statistics
    stats = monitor.get_statistics()
    print("Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Get health status
    health = monitor.get_health_status()
    print("\nHealth Status:")
    print(json.dumps(health, indent=2))
    
    # Export metrics
    monitor.export_metrics('api_metrics.json')