# Beverly Knits ERP v2 - Master Implementation Plan
**Document Version:** 1.0  
**Date:** December 2024  
**Status:** ACTIVE - Ready for Implementation  
**Priority:** Architectural Refactoring & Performance Optimization (Authentication Deferred)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Implementation Phases](#implementation-phases)
4. [Phase 1: Service Extraction & Integration](#phase-1-service-extraction--integration-weeks-1-2)
5. [Phase 2: Performance Optimization](#phase-2-performance-optimization-week-3)
6. [Phase 3: Feature Completion](#phase-3-feature-completion-week-4)
7. [Phase 4: Data Layer Refinement](#phase-4-data-layer-refinement-week-5)
8. [Phase 5: Testing & Quality](#phase-5-testing--quality-week-6)
9. [Phase 6: Infrastructure & Deployment](#phase-6-infrastructure--deployment-week-7)
10. [Phase 7: Documentation & Handoff](#phase-7-documentation--handoff-week-8)
11. [Technical Details](#technical-details)
12. [Risk Management](#risk-management)
13. [Success Metrics](#success-metrics)
14. [Immediate Next Steps](#immediate-next-steps)

---

## Executive Summary

### Project Overview
Beverly Knits ERP v2 requires comprehensive refactoring to transform from a monolithic 18,076-line application into a modular, maintainable, and high-performance system. This plan focuses on architectural improvements, performance optimization, and feature completion while deferring authentication implementation.

### Current Challenges
- **Monolithic Architecture**: 18,076-line single file with 127 API endpoints
- **Performance Issues**: 157 instances of DataFrame.iterrows() causing 10-100x slowdowns
- **Code Duplication**: 90+ repeated patterns across modules
- **Incomplete Features**: Fabric production API, alert system, cache warming
- **Technical Debt**: No service separation, mixed concerns, poor error handling

### Implementation Strategy
- **8-week phased approach** with parallel track execution
- **Zero-downtime migration** using feature flags and gradual rollout
- **Backward compatibility** maintained throughout
- **Performance-first** optimization delivering immediate business value

### Expected Outcomes
- **90% reduction** in monolith size (18,076 → <2,000 lines)
- **10-100x performance improvement** from vectorization
- **100% feature completion** including missing APIs
- **80% test coverage** with comprehensive test suite
- **50% reduction** in development time for new features

---

## Current State Assessment

### System Statistics
```yaml
Codebase Metrics:
  Total Python Files: 193
  Total Lines of Code: 91,036
  Main Monolith Size: 18,076 lines
  API Endpoints: 127 (45+ deprecated)
  Test Files: 37
  Code Duplication: 90+ instances

Data Volume:
  Yarn Items: 1,199
  BOM Entries: 28,653
  Production Orders: 194
  Work Centers: 91
  Machines: 285
  Total Production Load: 557,671 lbs

Performance Metrics:
  Data Load Time: 1-2 seconds (with cache)
  API Response: <200ms average (variable)
  Cache Hit Rate: 70-90%
  Dashboard Load: <3 seconds
```

### Completed vs Pending Tasks

#### ✅ Partially Completed (40% Overall)
- **Service Extraction**: 7 services created but NOT integrated
  - inventory_analyzer_service.py
  - sales_forecasting_service.py
  - capacity_planning_service.py
  - yarn_requirement_service.py
- **API v2 Structure**: Created but not connected
- **Data Layer**: Unified loader exists, connection pool created
- **Testing**: 37 test files exist but coverage unknown

#### ❌ Not Completed (60% Remaining)
- **Monolith Unchanged**: Still 18,076 lines with all routes
- **Performance Issues**: 157 iterrows() instances remain
- **Missing Features**: Fabric API, alerts, cache warming
- **No Integration**: Extracted services not wired up
- **No CI/CD Pipeline**: Manual deployment only

---

## Implementation Phases

### Phase Timeline Overview
```
Week 1-2: Service Extraction & Integration (Foundation)
Week 3:   Performance Optimization (Quick Wins)
Week 4:   Feature Completion (Business Value)
Week 5:   Data Layer Refinement (Stability)
Week 6:   Testing & Quality (Reliability)
Week 7:   Infrastructure & Deployment (Scalability)
Week 8:   Documentation & Handoff (Sustainability)
```

---

## Phase 1: Service Extraction & Integration (Weeks 1-2)

### Objective
Extract all business logic from monolith and integrate existing services, reducing main file from 18,076 to <2,000 lines.

### Week 1: Complete Service Extraction

#### Day 1-2: Extract Remaining Core Services
```python
# Extract these classes from beverly_comprehensive_erp.py:
# Lines 8000-9000 → src/services/production_scheduler_service.py
class ProductionSchedulerService:
    def __init__(self, repo, capacity_engine):
        self.repo = repo
        self.capacity_engine = capacity_engine
    
    def schedule_production(self, orders):
        # Extract scheduling logic
        pass

# Lines 12000-13000 → src/services/manufacturing_supply_chain_service.py
class ManufacturingSupplyChainService:
    def __init__(self, inventory, forecasting, planning):
        self.inventory = inventory
        self.forecasting = forecasting
        self.planning = planning

# Lines 14000-15000 → src/services/time_phased_mrp_service.py
class TimePhasedMRPService:
    def calculate_requirements(self, demand, lead_times):
        # Extract MRP logic
        pass
```

#### Day 3-4: Wire Up Already-Extracted Services
```python
# In beverly_comprehensive_erp.py, replace embedded code:
from src.services.inventory_analyzer_service import InventoryAnalyzerService
from src.services.sales_forecasting_service import SalesForecastingService

# Initialize with dependency injection
def create_app():
    # Service initialization
    data_loader = UnifiedDataLoader(config)
    
    services = {
        'inventory': InventoryAnalyzerService(data_loader),
        'forecasting': SalesForecastingService(ml_config),
        'capacity': CapacityPlanningService(data_loader),
        'yarn': YarnRequirementService(data_loader)
    }
    
    # Replace inline code with service calls
    @app.route('/api/inventory-intelligence-enhanced')
    def get_inventory():
        return services['inventory'].analyze()
```

#### Day 5: Service Registry Implementation
```python
# src/services/service_container.py
class ServiceContainer:
    """Centralized service management"""
    
    def __init__(self):
        self._services = {}
        self._initialize_services()
    
    def _initialize_services(self):
        # Core services
        self.register('data_loader', UnifiedDataLoader())
        self.register('cache', CacheManager())
        
        # Business services
        self.register('inventory', InventoryAnalyzerService(
            self.get('data_loader'),
            self.get('cache')
        ))
        
    def get(self, name):
        return self._services.get(name)
    
    def register(self, name, service):
        self._services[name] = service

# Global service container
services = ServiceContainer()
```

### Week 2: API Migration to v2

#### Day 1-2: Migrate All 127 Endpoints
```python
# src/api/v2/routes.py
from flask import Blueprint

# Create modular blueprints
inventory_bp = Blueprint('inventory', __name__)
production_bp = Blueprint('production', __name__)
yarn_bp = Blueprint('yarn', __name__)
forecasting_bp = Blueprint('forecasting', __name__)

# Move routes from monolith
@inventory_bp.route('/inventory')
def get_inventory():
    service = services.get('inventory')
    return service.get_enhanced_intelligence(
        view=request.args.get('view', 'summary'),
        realtime=request.args.get('realtime', False)
    )

# Register all blueprints
def register_blueprints(app):
    app.register_blueprint(inventory_bp, url_prefix='/api/v2')
    app.register_blueprint(production_bp, url_prefix='/api/v2')
    app.register_blueprint(yarn_bp, url_prefix='/api/v2')
    app.register_blueprint(forecasting_bp, url_prefix='/api/v2')
```

#### Day 3-4: Implement Backward Compatibility
```python
# src/api/compatibility.py
def create_legacy_redirects(app):
    """Maintain backward compatibility for deprecated endpoints"""
    
    legacy_mappings = {
        '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
        '/api/production-status': '/api/v2/production/status',
        '/api/forecast-demand': '/api/v2/forecasting/demand',
        # ... 42+ more mappings
    }
    
    for old_path, new_path in legacy_mappings.items():
        @app.route(old_path)
        def redirect_legacy():
            return redirect(new_path, code=307)
```

#### Day 5: Validate All Endpoints
```python
# tests/integration/test_api_migration.py
def test_all_endpoints_working():
    """Ensure all 127 endpoints respond correctly"""
    
    endpoints = load_endpoint_registry()
    
    for endpoint in endpoints:
        response = client.get(endpoint['path'])
        assert response.status_code in [200, 307]  # OK or redirect
        
        if response.status_code == 307:
            # Follow redirect
            response = client.get(response.headers['Location'])
            assert response.status_code == 200
```

---

## Phase 2: Performance Optimization (Week 3)

### Objective
Achieve 10-100x performance improvements through vectorization and optimization.

### Day 1: Eliminate DataFrame.iterrows() - Priority Files
```python
# CRITICAL: Fix these files first (highest impact)
# beverly_comprehensive_erp.py - 59 instances
# six_phase_planning_engine.py - 10 instances
# database_etl_pipeline.py - 9 instances

# src/utils/dataframe_optimizer.py
class DataFrameOptimizer:
    @staticmethod
    def optimize_iterrows(df, operation):
        """Replace iterrows with vectorized operations"""
        
        # BEFORE (Slow - O(n)):
        # for index, row in df.iterrows():
        #     df.at[index, 'result'] = calculate(row['value'])
        
        # AFTER (Fast - O(1) vectorized):
        df['result'] = df['value'].apply(operation)
        # Or for maximum performance:
        df['result'] = np.vectorize(operation)(df['value'].values)
        
        return df

# Automated optimization script
def auto_optimize_file(filepath):
    """Automatically replace iterrows patterns"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match iterrows
    pattern = r'for\s+\w+,\s*\w+\s+in\s+(\w+)\.iterrows\(\):'
    
    # Replace with vectorized version
    optimized = re.sub(pattern, lambda m: f"# Vectorized\n{m.group(1)}.apply(lambda row:", content)
    
    with open(filepath, 'w') as f:
        f.write(optimized)
```

### Day 2: Implement Batch Processing
```python
# src/utils/batch_processor.py
from typing import Callable, Any
import pandas as pd

class BatchProcessor:
    """Process large datasets in optimized batches"""
    
    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
        self.metrics = BatchMetrics()
    
    def process_dataframe(self, 
                          df: pd.DataFrame, 
                          processor: Callable,
                          parallel: bool = True) -> pd.DataFrame:
        """Process DataFrame in batches with optional parallelization"""
        
        if len(df) <= self.batch_size:
            return processor(df)
        
        results = []
        
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = []
                for start in range(0, len(df), self.batch_size):
                    batch = df.iloc[start:start + self.batch_size]
                    futures.append(executor.submit(processor, batch))
                
                for future in futures:
                    results.append(future.result())
        else:
            for start in range(0, len(df), self.batch_size):
                batch = df.iloc[start:start + self.batch_size]
                results.append(processor(batch))
        
        return pd.concat(results, ignore_index=True)
```

### Day 3: Database Query Optimization
```python
# src/database/query_optimizer.py
class QueryOptimizer:
    """Optimize database queries for performance"""
    
    @staticmethod
    def batch_fetch(ids: List[str], table: str, batch_size: int = 1000):
        """Fetch records in batches to avoid N+1 queries"""
        
        # BEFORE: N+1 Problem
        # for order_id in order_ids:
        #     yarn = db.query(f"SELECT * FROM yarns WHERE order_id = {order_id}")
        
        # AFTER: Single optimized query
        results = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch_ids))
            query = f"SELECT * FROM {table} WHERE id IN ({placeholders})"
            results.extend(db.execute(query, batch_ids))
        
        return results
    
    @staticmethod
    def add_indexes(connection):
        """Add missing database indexes"""
        indexes = [
            "CREATE INDEX idx_yarn_planning ON yarn_inventory(planning_balance)",
            "CREATE INDEX idx_order_status ON production_orders(status)",
            "CREATE INDEX idx_bom_style ON bom(style_id, yarn_id)",
            "CREATE INDEX idx_machine_center ON machines(work_center_id)"
        ]
        
        for index in indexes:
            try:
                connection.execute(index)
            except:
                pass  # Index may already exist
```

### Day 4: Memory Optimization
```python
# src/utils/memory_optimizer.py
import pandas as pd
import numpy as np

class MemoryOptimizer:
    """Reduce DataFrame memory usage by 50-90%"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert strings to categories (huge savings)
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        # Convert datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory/initial_memory) * 100
        
        print(f"Memory reduced by {reduction:.1f}% ({initial_memory:.1f}MB → {final_memory:.1f}MB)")
        
        return df
```

### Day 5: Caching Strategy Implementation
```python
# src/cache/multi_tier_cache.py
import json
import pickle
from typing import Any, Optional, Callable
import redis
import hashlib

class MultiTierCache:
    """Multi-level caching for optimal performance"""
    
    def __init__(self):
        self.l1_memory = {}  # In-memory (microseconds)
        self.l1_max_size = 100  # Limit memory cache size
        self.l2_redis = redis.Redis(host='localhost', port=6379)
        self.ttls = {
            'yarn_inventory': 300,  # 5 minutes
            'bom_data': 3600,  # 1 hour
            'production_orders': 60,  # 1 minute
            'ml_predictions': 1800  # 30 minutes
        }
    
    def _get_ttl(self, key: str) -> int:
        """Get TTL based on data type"""
        for data_type, ttl in self.ttls.items():
            if data_type in key:
                return ttl
        return 300  # Default 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback chain"""
        
        # L1: Memory cache (fastest)
        if key in self.l1_memory:
            return self.l1_memory[key]
        
        # L2: Redis cache
        value = self.l2_redis.get(key)
        if value:
            deserialized = pickle.loads(value)
            # Promote to L1
            self._promote_to_l1(key, deserialized)
            return deserialized
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set in all cache levels"""
        
        ttl = ttl or self._get_ttl(key)
        
        # L1: Memory cache
        self._promote_to_l1(key, value)
        
        # L2: Redis cache
        serialized = pickle.dumps(value)
        self.l2_redis.setex(key, ttl, serialized)
    
    def _promote_to_l1(self, key: str, value: Any):
        """Add to L1 cache with LRU eviction"""
        if len(self.l1_memory) >= self.l1_max_size:
            # Remove oldest item (simple LRU)
            oldest = next(iter(self.l1_memory))
            del self.l1_memory[oldest]
        
        self.l1_memory[key] = value
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        
        # Clear L1
        keys_to_remove = [k for k in self.l1_memory if pattern in k]
        for key in keys_to_remove:
            del self.l1_memory[key]
        
        # Clear L2
        for key in self.l2_redis.scan_iter(match=f"*{pattern}*"):
            self.l2_redis.delete(key)
```

---

## Phase 3: Feature Completion (Week 4)

### Objective
Complete all missing features and fix known issues to achieve 100% functionality.

### Day 1: Implement Fabric Production API
```python
# src/api/v2/fabric_production.py
from flask import Blueprint, jsonify, request
from src.services.fabric_analyzer import FabricProductionAnalyzer

fabric_bp = Blueprint('fabric', __name__)

@fabric_bp.route('/fabric-production', methods=['GET'])
def get_fabric_production():
    """Complete fabric production API implementation"""
    
    analyzer = FabricProductionAnalyzer(services.get('data_loader'))
    
    # Get production data
    production_data = analyzer.analyze_production(
        start_date=request.args.get('start_date'),
        end_date=request.args.get('end_date'),
        style_filter=request.args.get('style')
    )
    
    # Get demand analysis
    demand_data = analyzer.analyze_demand(
        horizon_days=int(request.args.get('horizon', 30))
    )
    
    # Get capacity utilization
    capacity_data = analyzer.get_capacity_utilization()
    
    return jsonify({
        'status': 'success',
        'production': {
            'current_orders': production_data['orders'],
            'in_progress': production_data['in_progress'],
            'completed': production_data['completed'],
            'efficiency': production_data['efficiency']
        },
        'demand': {
            'forecast': demand_data['forecast'],
            'confirmed_orders': demand_data['confirmed'],
            'coverage_days': demand_data['coverage']
        },
        'capacity': {
            'utilization_percent': capacity_data['utilization'],
            'available_hours': capacity_data['available'],
            'bottlenecks': capacity_data['bottlenecks']
        }
    })

# src/services/fabric_analyzer.py
class FabricProductionAnalyzer:
    """Analyze fabric production and demand"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cache = MultiTierCache()
    
    def analyze_production(self, start_date=None, end_date=None, style_filter=None):
        """Analyze production orders and status"""
        
        # Check cache
        cache_key = f"fabric_production:{start_date}:{end_date}:{style_filter}"
        if cached := self.cache.get(cache_key):
            return cached
        
        # Load production data
        orders_df = self.data_loader.load_production_orders()
        
        # Apply filters
        if start_date:
            orders_df = orders_df[orders_df['scheduled_date'] >= start_date]
        if end_date:
            orders_df = orders_df[orders_df['scheduled_date'] <= end_date]
        if style_filter:
            orders_df = orders_df[orders_df['style_id'] == style_filter]
        
        # Calculate metrics
        result = {
            'orders': len(orders_df),
            'in_progress': len(orders_df[orders_df['status'] == 'in_progress']),
            'completed': len(orders_df[orders_df['status'] == 'completed']),
            'efficiency': self._calculate_efficiency(orders_df)
        }
        
        # Cache result
        self.cache.set(cache_key, result, ttl=300)
        
        return result
```

### Day 2: Complete Alert System
```python
# src/monitoring/alert_manager.py
import smtplib
from email.mime.text import MIMEText
import requests
from typing import List, Dict, Any

class AlertManager:
    """Complete alert system with multiple channels"""
    
    def __init__(self, config):
        self.smtp_config = config.get('smtp', {})
        self.webhook_urls = config.get('webhooks', {})
        self.sms_config = config.get('sms', {})
        self.alert_rules = self._load_alert_rules()
    
    def _load_alert_rules(self):
        """Load alert rules and thresholds"""
        return {
            'yarn_shortage_critical': {
                'condition': lambda data: data['shortage_count'] > 10,
                'severity': 'CRITICAL',
                'channels': ['email', 'sms', 'webhook']
            },
            'capacity_overload': {
                'condition': lambda data: data['utilization'] > 95,
                'severity': 'HIGH',
                'channels': ['email', 'webhook']
            },
            'forecast_deviation': {
                'condition': lambda data: data['deviation'] > 20,
                'severity': 'MEDIUM',
                'channels': ['email']
            }
        }
    
    def check_and_alert(self, metric_name: str, data: Dict[str, Any]):
        """Check metrics against rules and send alerts"""
        
        if metric_name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric_name]
        
        if rule['condition'](data):
            self.send_alert(
                title=f"Alert: {metric_name}",
                message=self._format_message(metric_name, data),
                severity=rule['severity'],
                channels=rule['channels']
            )
    
    def send_alert(self, title: str, message: str, severity: str, channels: List[str]):
        """Send alert through specified channels"""
        
        for channel in channels:
            try:
                if channel == 'email':
                    self._send_email(title, message, severity)
                elif channel == 'webhook':
                    self._send_webhook(title, message, severity)
                elif channel == 'sms':
                    self._send_sms(message, severity)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_email(self, subject: str, body: str, severity: str):
        """Send email alert"""
        
        msg = MIMEText(body)
        msg['Subject'] = f"[{severity}] {subject}"
        msg['From'] = self.smtp_config['from']
        msg['To'] = ', '.join(self._get_recipients(severity))
        
        with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
    
    def _send_webhook(self, title: str, message: str, severity: str):
        """Send webhook notification (Slack/Teams/Discord)"""
        
        # Slack format
        payload = {
            'text': title,
            'attachments': [{
                'color': self._severity_to_color(severity),
                'text': message,
                'footer': 'Beverly Knits ERP',
                'ts': int(time.time())
            }]
        }
        
        webhook_url = self.webhook_urls.get(severity.lower(), self.webhook_urls.get('default'))
        if webhook_url:
            requests.post(webhook_url, json=payload)
    
    def _send_sms(self, message: str, severity: str):
        """Send SMS alert for critical issues"""
        
        if severity != 'CRITICAL':
            return
        
        # Using Twilio as example
        from twilio.rest import Client
        
        client = Client(self.sms_config['account_sid'], self.sms_config['auth_token'])
        
        for recipient in self.sms_config['recipients']:
            client.messages.create(
                body=f"[{severity}] {message[:160]}",  # SMS limit
                from_=self.sms_config['from_number'],
                to=recipient
            )
```

### Day 3: Implement Cache Warming
```python
# src/cache/cache_warmer.py
import asyncio
from typing import List, Dict, Callable
import schedule

class CacheWarmer:
    """Proactive cache warming for frequently accessed data"""
    
    def __init__(self, cache: MultiTierCache, data_loader):
        self.cache = cache
        self.data_loader = data_loader
        self.warming_tasks = self._define_warming_tasks()
        
    def _define_warming_tasks(self) -> List[Dict]:
        """Define what data to warm and when"""
        return [
            {
                'name': 'yarn_inventory',
                'loader': self.data_loader.load_yarn_inventory,
                'cache_key': 'yarn_inventory:all',
                'schedule': 'every 5 minutes',
                'priority': 'HIGH'
            },
            {
                'name': 'bom_data',
                'loader': self.data_loader.load_bom_data,
                'cache_key': 'bom:all',
                'schedule': 'every hour',
                'priority': 'MEDIUM'
            },
            {
                'name': 'production_orders',
                'loader': self.data_loader.load_production_orders,
                'cache_key': 'production:active',
                'schedule': 'every 2 minutes',
                'priority': 'HIGH'
            },
            {
                'name': 'ml_forecasts',
                'loader': self._generate_forecasts,
                'cache_key': 'forecast:latest',
                'schedule': 'every 30 minutes',
                'priority': 'LOW'
            }
        ]
    
    def warm_cache_on_startup(self):
        """Warm cache with critical data on application startup"""
        
        print("Starting cache warming...")
        start_time = time.time()
        
        # Sort by priority
        tasks = sorted(self.warming_tasks, 
                      key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']])
        
        for task in tasks:
            try:
                print(f"Warming {task['name']}...")
                data = task['loader']()
                self.cache.set(task['cache_key'], data)
                print(f"✓ {task['name']} warmed")
            except Exception as e:
                logger.error(f"Failed to warm {task['name']}: {e}")
        
        elapsed = time.time() - start_time
        print(f"Cache warming completed in {elapsed:.2f} seconds")
    
    def start_scheduled_warming(self):
        """Start background cache warming scheduler"""
        
        for task in self.warming_tasks:
            schedule_time = task['schedule']
            
            if 'every' in schedule_time:
                parts = schedule_time.split()
                interval = int(parts[1])
                unit = parts[2]
                
                if unit == 'minutes':
                    schedule.every(interval).minutes.do(
                        self._warm_task, task
                    )
                elif unit == 'hour' or unit == 'hours':
                    schedule.every(interval).hours.do(
                        self._warm_task, task
                    )
        
        # Run scheduler in background thread
        import threading
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
    
    def _warm_task(self, task: Dict):
        """Warm a specific cache task"""
        try:
            data = task['loader']()
            self.cache.set(task['cache_key'], data)
            logger.debug(f"Cache warmed: {task['name']}")
        except Exception as e:
            logger.error(f"Cache warming failed for {task['name']}: {e}")
```

### Day 4: Add Real-time Updates (WebSocket)
```python
# src/realtime/websocket_server.py
from flask_socketio import SocketIO, emit, join_room, leave_room
import json

# Initialize SocketIO
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

class RealtimeUpdateManager:
    """Manage real-time updates via WebSocket"""
    
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
        self.subscriptions = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('subscribe')
        def handle_subscription(data):
            """Handle subscription requests"""
            room = data.get('channel')
            if room:
                join_room(room)
                self.subscriptions[request.sid] = room
                emit('subscribed', {'channel': room, 'status': 'success'})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscribe requests"""
            room = data.get('channel')
            if room:
                leave_room(room)
                if request.sid in self.subscriptions:
                    del self.subscriptions[request.sid]
                emit('unsubscribed', {'channel': room})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Clean up on disconnect"""
            if request.sid in self.subscriptions:
                del self.subscriptions[request.sid]
    
    def broadcast_update(self, channel: str, data: Dict):
        """Broadcast update to all subscribers of a channel"""
        
        self.socketio.emit('update', {
            'channel': channel,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }, room=channel)
    
    def send_inventory_update(self, yarn_id: str, new_balance: float):
        """Send inventory update"""
        self.broadcast_update(f'inventory:{yarn_id}', {
            'yarn_id': yarn_id,
            'planning_balance': new_balance,
            'status': 'critical' if new_balance < 0 else 'normal'
        })
    
    def send_production_update(self, order_id: str, status: str):
        """Send production status update"""
        self.broadcast_update('production:updates', {
            'order_id': order_id,
            'status': status,
            'updated_at': datetime.utcnow().isoformat()
        })

# Integration with existing services
class InventoryService:
    def update_planning_balance(self, yarn_id: str, new_balance: float):
        # Update database
        self.repo.update_balance(yarn_id, new_balance)
        
        # Send real-time update
        realtime_manager.send_inventory_update(yarn_id, new_balance)
```

### Day 5: Complete ML Model Endpoints
```python
# src/api/v2/ml_forecasting.py
from flask import Blueprint, jsonify, request
import numpy as np

ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/forecast/ensemble', methods=['POST'])
def ensemble_forecast():
    """Complete ML ensemble forecasting endpoint"""
    
    # Validate input
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({'error': 'Missing items for forecast'}), 400
    
    # Get forecast parameters
    items = data['items']
    horizon = data.get('horizon', 30)
    confidence_level = data.get('confidence', 0.95)
    
    # Load models
    models = {
        'arima': load_model('arima'),
        'prophet': load_model('prophet'),
        'lstm': load_model('lstm'),
        'xgboost': load_model('xgboost')
    }
    
    # Generate predictions from each model
    predictions = {}
    weights = {'arima': 0.2, 'prophet': 0.25, 'lstm': 0.35, 'xgboost': 0.2}
    
    for item in items:
        item_predictions = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(item, horizon)
                item_predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Model {model_name} failed for {item}: {e}")
                item_predictions[model_name] = None
        
        # Ensemble combination
        valid_preds = [p for p in item_predictions.values() if p is not None]
        
        if valid_preds:
            # Weighted average
            ensemble_pred = sum(
                weights[name] * pred 
                for name, pred in item_predictions.items() 
                if pred is not None
            ) / sum(weights[name] for name, pred in item_predictions.items() if pred is not None)
            
            # Calculate confidence intervals
            std_dev = np.std(valid_preds)
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            
            predictions[item] = {
                'forecast': ensemble_pred.tolist(),
                'confidence_lower': (ensemble_pred - z_score * std_dev).tolist(),
                'confidence_upper': (ensemble_pred + z_score * std_dev).tolist(),
                'models_used': [n for n, p in item_predictions.items() if p is not None],
                'horizon': horizon
            }
        else:
            predictions[item] = {
                'error': 'All models failed',
                'fallback': 'historical_average'
            }
    
    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'metadata': {
            'ensemble_method': 'weighted_average',
            'weights': weights,
            'confidence_level': confidence_level,
            'generated_at': datetime.utcnow().isoformat()
        }
    })

@ml_bp.route('/forecast/retrain', methods=['POST'])
def retrain_models():
    """Trigger model retraining"""
    
    # This would typically be an async task
    from src.ml.training_pipeline import ModelTrainingPipeline
    
    pipeline = ModelTrainingPipeline()
    
    # Start training in background
    import threading
    def train():
        results = pipeline.retrain_all_models()
        # Store results
        cache.set('ml:training_results', results, ttl=86400)
    
    thread = threading.Thread(target=train, daemon=True)
    thread.start()
    
    return jsonify({
        'status': 'training_started',
        'message': 'Model retraining initiated in background',
        'check_status_at': '/api/v2/ml/training-status'
    })
```

---

## Phase 4: Data Layer Refinement (Week 5)

### Objective
Implement clean data access patterns and unify data handling across the system.

### Day 1-2: Repository Pattern Implementation
```python
# src/repositories/base_repository.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict
import pandas as pd

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Base repository for data access"""
    
    def __init__(self, db_connection, cache):
        self.db = db_connection
        self.cache = cache
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[T]:
        pass
    
    @abstractmethod
    def get_all(self, filters: Dict = None) -> List[T]:
        pass
    
    @abstractmethod
    def save(self, entity: T) -> T:
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

# src/repositories/yarn_repository.py
class YarnRepository(BaseRepository):
    """Repository for yarn data access"""
    
    def get_by_id(self, yarn_id: str) -> Optional[Yarn]:
        """Get yarn by ID with caching"""
        
        # Check cache
        cache_key = f"yarn:{yarn_id}"
        if cached := self.cache.get(cache_key):
            return Yarn.from_dict(cached)
        
        # Query database
        query = "SELECT * FROM yarn_inventory WHERE yarn_id = ?"
        result = self.db.fetch_one(query, (yarn_id,))
        
        if result:
            yarn = Yarn.from_db(result)
            self.cache.set(cache_key, yarn.to_dict(), ttl=300)
            return yarn
        
        return None
    
    def get_shortages(self, threshold: float = 0) -> pd.DataFrame:
        """Get all yarns with planning balance below threshold"""
        
        cache_key = f"yarn:shortages:{threshold}"
        if cached := self.cache.get(cache_key):
            return pd.DataFrame(cached)
        
        query = """
            SELECT yi.*, 
                   yi.theoretical_balance + yi.allocated + yi.on_order as planning_balance
            FROM yarn_inventory yi
            WHERE (yi.theoretical_balance + yi.allocated + yi.on_order) < ?
            ORDER BY planning_balance ASC
        """
        
        df = pd.read_sql(query, self.db.connection, params=(threshold,))
        self.cache.set(cache_key, df.to_dict('records'), ttl=60)
        
        return df
    
    def update_balance(self, yarn_id: str, new_balance: float) -> bool:
        """Update yarn planning balance"""
        
        query = """
            UPDATE yarn_inventory 
            SET theoretical_balance = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE yarn_id = ?
        """
        
        self.db.execute(query, (new_balance, yarn_id))
        
        # Invalidate cache
        self.cache.invalidate(f"yarn:{yarn_id}")
        self.cache.invalidate("yarn:shortages")
        
        return True
```

### Day 3: Column Mapping Unification
```python
# src/utils/column_mapper.py
from typing import Dict, List, Optional
import pandas as pd

class ColumnMapper:
    """Unified column mapping for consistent data handling"""
    
    # Master mapping configuration
    COLUMN_MAPPINGS = {
        'planning_balance': {
            'variations': ['Planning Balance', 'Planning_Balance', 'planning balance'],
            'standard': 'planning_balance',
            'type': 'float'
        },
        'yarn_id': {
            'variations': ['Desc#', 'desc_num', 'YarnID', 'yarn_id', 'Yarn ID'],
            'standard': 'yarn_id',
            'type': 'str'
        },
        'style': {
            'variations': ['fStyle#', 'Style#', 'style_num', 'Style'],
            'standard': 'style_id',
            'type': 'str'
        },
        'quantity': {
            'variations': ['Qty', 'Quantity', 'quantity', 'Amount'],
            'standard': 'quantity',
            'type': 'float'
        },
        'balance': {
            'variations': ['Balance (lbs)', 'Balance', 'balance_lbs'],
            'standard': 'balance',
            'type': 'float'
        }
    }
    
    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize all column names in DataFrame"""
        
        standardized_df = df.copy()
        
        for standard_name, config in cls.COLUMN_MAPPINGS.items():
            # Find matching column
            for col in df.columns:
                if col in config['variations']:
                    # Rename column
                    standardized_df.rename(
                        columns={col: config['standard']}, 
                        inplace=True
                    )
                    
                    # Convert type if needed
                    if config['type'] == 'float':
                        # Handle comma-separated numbers
                        if standardized_df[config['standard']].dtype == 'object':
                            standardized_df[config['standard']] = (
                                standardized_df[config['standard']]
                                .str.replace(',', '')
                                .str.replace('$', '')
                                .astype(float)
                            )
                    
                    break
        
        return standardized_df
    
    @classmethod
    def find_column(cls, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """Find column by type, trying all variations"""
        
        if column_type not in cls.COLUMN_MAPPINGS:
            return None
        
        variations = cls.COLUMN_MAPPINGS[column_type]['variations']
        
        for col in df.columns:
            if col in variations:
                return col
        
        return None
    
    @classmethod
    def validate_required_columns(cls, df: pd.DataFrame, required: List[str]) -> Dict:
        """Validate DataFrame has required columns"""
        
        result = {
            'valid': True,
            'missing': [],
            'found': {}
        }
        
        for req_col in required:
            col_name = cls.find_column(df, req_col)
            if col_name:
                result['found'][req_col] = col_name
            else:
                result['missing'].append(req_col)
                result['valid'] = False
        
        return result
```

### Day 4-5: Database Migration & Schema
```python
# alembic/versions/001_initial_schema.py
"""Initial database schema

Revision ID: 001
Create Date: 2024-12-01
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    """Create initial database schema"""
    
    # Yarn master data
    op.create_table('yarns',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('yarn_id', sa.String(50), unique=True, nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('category', sa.String(50)),
        sa.Column('supplier', sa.String(100)),
        sa.Column('lead_time_days', sa.Integer),
        sa.Column('min_order_qty', sa.Numeric(10, 2)),
        sa.Column('unit_cost', sa.Numeric(10, 2)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )
    
    # Yarn inventory
    op.create_table('yarn_inventory',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('yarn_id', sa.String(50), sa.ForeignKey('yarns.yarn_id')),
        sa.Column('theoretical_balance', sa.Numeric(12, 2)),
        sa.Column('allocated', sa.Numeric(12, 2)),  # Stored as negative
        sa.Column('on_order', sa.Numeric(12, 2)),
        sa.Column('planning_balance', sa.Numeric(12, 2)),  # Computed column
        sa.Column('last_updated', sa.DateTime, server_default=sa.func.now())
    )
    
    # Bill of Materials
    op.create_table('bom',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('style_id', sa.String(50), nullable=False),
        sa.Column('yarn_id', sa.String(50), sa.ForeignKey('yarns.yarn_id')),
        sa.Column('quantity_per_unit', sa.Numeric(10, 4)),
        sa.Column('unit_of_measure', sa.String(10)),
        sa.Column('active', sa.Boolean, server_default='true')
    )
    
    # Production orders
    op.create_table('production_orders',
        sa.Column('order_id', sa.String(50), primary_key=True),
        sa.Column('style_id', sa.String(50)),
        sa.Column('quantity', sa.Numeric(10, 2)),
        sa.Column('machine_id', sa.Integer),
        sa.Column('work_center', sa.String(20)),
        sa.Column('status', sa.String(20)),
        sa.Column('scheduled_date', sa.Date),
        sa.Column('completion_date', sa.Date),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
    
    # Create indexes for performance
    op.create_index('idx_yarn_planning', 'yarn_inventory', ['planning_balance'])
    op.create_index('idx_order_status', 'production_orders', ['status'])
    op.create_index('idx_bom_style', 'bom', ['style_id', 'yarn_id'])
    op.create_index('idx_order_scheduled', 'production_orders', ['scheduled_date'])
    
    # Create views for common queries
    op.execute("""
        CREATE VIEW v_yarn_shortages AS
        SELECT 
            yi.*,
            y.description,
            y.supplier,
            y.lead_time_days
        FROM yarn_inventory yi
        JOIN yarns y ON yi.yarn_id = y.yarn_id
        WHERE yi.planning_balance < 0
        ORDER BY yi.planning_balance ASC
    """)
    
    op.execute("""
        CREATE VIEW v_production_capacity AS
        SELECT 
            work_center,
            COUNT(DISTINCT machine_id) as machine_count,
            SUM(CASE WHEN status = 'in_progress' THEN quantity ELSE 0 END) as active_load,
            SUM(CASE WHEN status = 'scheduled' THEN quantity ELSE 0 END) as scheduled_load
        FROM production_orders
        WHERE status IN ('in_progress', 'scheduled')
        GROUP BY work_center
    """)

def downgrade():
    """Drop all tables"""
    op.drop_table('production_orders')
    op.drop_table('bom')
    op.drop_table('yarn_inventory')
    op.drop_table('yarns')
```

---

## Phase 5: Testing & Quality (Week 6)

### Objective
Achieve 80% test coverage and establish comprehensive testing practices.

### Day 1-2: Unit Testing
```python
# tests/unit/test_inventory_service.py
import pytest
from unittest.mock import Mock, patch
from src.services.inventory_analyzer_service import InventoryAnalyzerService
from src.models.yarn import Yarn

class TestInventoryAnalyzerService:
    
    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies"""
        mock_repo = Mock()
        mock_cache = Mock()
        return InventoryAnalyzerService(mock_repo, mock_cache)
    
    def test_calculate_planning_balance_positive(self, service):
        """Test planning balance calculation with positive result"""
        
        # Arrange
        service.repository.get_by_id.return_value = Yarn(
            yarn_id="Y001",
            theoretical_balance=100.0,
            allocated=-20.0,  # Negative allocated
            on_order=50.0
        )
        
        # Act
        balance = service.calculate_planning_balance("Y001")
        
        # Assert
        assert balance == 130.0  # 100 + (-20) + 50
        service.repository.get_by_id.assert_called_once_with("Y001")
    
    def test_shortage_detection(self, service):
        """Test yarn shortage detection"""
        
        # Arrange
        mock_data = pd.DataFrame([
            {'yarn_id': 'Y001', 'planning_balance': -100},
            {'yarn_id': 'Y002', 'planning_balance': 50},
            {'yarn_id': 'Y003', 'planning_balance': -25}
        ])
        
        service.repository.get_all.return_value = mock_data
        
        # Act
        shortages = service.detect_shortages(threshold=0)
        
        # Assert
        assert len(shortages) == 2
        assert 'Y001' in shortages['yarn_id'].values
        assert 'Y003' in shortages['yarn_id'].values
        assert 'Y002' not in shortages['yarn_id'].values
    
    def test_cache_usage(self, service):
        """Test that cache is properly used"""
        
        # Arrange
        cache_key = "planning_balance:Y001"
        service.cache.get.return_value = 130.0
        
        # Act
        balance = service.calculate_planning_balance("Y001")
        
        # Assert
        assert balance == 130.0
        service.cache.get.assert_called_once_with(cache_key)
        service.repository.get_by_id.assert_not_called()  # Should not hit DB
```

### Day 3: Integration Testing
```python
# tests/integration/test_api_integration.py
import pytest
from src.app import create_app
import json

class TestAPIIntegration:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app = create_app(testing=True)
        with app.test_client() as client:
            yield client
    
    def test_inventory_endpoint_integration(self, client):
        """Test inventory endpoint with real service integration"""
        
        response = client.get('/api/v2/inventory?view=summary')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'total_items' in data
        assert 'shortage_count' in data
        assert 'critical_shortages' in data
    
    def test_production_planning_workflow(self, client):
        """Test complete production planning workflow"""
        
        # Step 1: Get current inventory
        inv_response = client.get('/api/v2/inventory')
        assert inv_response.status_code == 200
        
        # Step 2: Check production capacity
        cap_response = client.get('/api/v2/production/capacity')
        assert cap_response.status_code == 200
        capacity = json.loads(cap_response.data)
        
        # Step 3: Create production order if capacity available
        if capacity['available_capacity'] > 0:
            order_data = {
                'style_id': 'S001',
                'quantity': 100,
                'priority': 'high'
            }
            
            order_response = client.post('/api/v2/production/orders',
                                        json=order_data)
            assert order_response.status_code in [201, 200]
    
    def test_backward_compatibility(self, client):
        """Test that deprecated endpoints still work"""
        
        # Old endpoint
        old_response = client.get('/api/yarn-inventory')
        
        # Should redirect to new endpoint
        assert old_response.status_code in [200, 307]
        
        if old_response.status_code == 307:
            # Follow redirect
            new_location = old_response.headers['Location']
            assert '/api/v2/' in new_location
```

### Day 4: Performance Testing
```python
# tests/performance/test_performance.py
import pytest
import time
import concurrent.futures
from locust import HttpUser, task, between

class TestPerformance:
    
    def test_response_time_under_threshold(self, client):
        """Ensure API responses are under 200ms"""
        
        endpoints = [
            '/api/v2/inventory',
            '/api/v2/production/status',
            '/api/v2/yarn/shortages',
            '/api/v2/forecasting/demand'
        ]
        
        for endpoint in endpoints:
            start = time.perf_counter()
            response = client.get(endpoint)
            duration = (time.perf_counter() - start) * 1000
            
            assert response.status_code == 200
            assert duration < 200, f"{endpoint} took {duration:.2f}ms"
    
    def test_concurrent_load(self, client):
        """Test system under concurrent load"""
        
        def make_request():
            return client.get('/api/v2/inventory')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r.status_code == 200)
        success_rate = success_count / len(results)
        
        assert success_rate >= 0.99, f"Success rate: {success_rate:.2%}"
    
    def test_dataframe_optimization_performance(self):
        """Test DataFrame operation performance improvements"""
        
        import pandas as pd
        import numpy as np
        
        # Create large test DataFrame
        df = pd.DataFrame({
            'yarn_id': [f'Y{i:04d}' for i in range(10000)],
            'value': np.random.randn(10000)
        })
        
        # Test vectorized operation (should be fast)
        start = time.perf_counter()
        df['result'] = df['value'].apply(lambda x: x * 2)
        vectorized_time = time.perf_counter() - start
        
        # Should complete in under 100ms for 10k rows
        assert vectorized_time < 0.1, f"Vectorized operation took {vectorized_time:.3f}s"

# Locust load testing
class ERPUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def view_inventory(self):
        self.client.get("/api/v2/inventory")
    
    @task(2)
    def check_production(self):
        self.client.get("/api/v2/production/status")
    
    @task(1)
    def get_forecast(self):
        self.client.get("/api/v2/forecasting/demand")
```

### Day 5: End-to-End Testing
```python
# tests/e2e/test_e2e_workflows.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestE2EWorkflows:
    
    @pytest.fixture
    def driver(self):
        """Create Selenium WebDriver"""
        driver = webdriver.Chrome()
        yield driver
        driver.quit()
    
    def test_complete_inventory_workflow(self, driver, base_url):
        """Test complete inventory management workflow"""
        
        # Navigate to dashboard
        driver.get(f"{base_url}/dashboard")
        
        # Check inventory tab
        inventory_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "inventory-tab"))
        )
        inventory_tab.click()
        
        # Wait for data to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "inventory-table"))
        )
        
        # Check for shortage alerts
        shortage_alerts = driver.find_elements(By.CLASS_NAME, "shortage-alert")
        
        if shortage_alerts:
            # Click on first shortage
            shortage_alerts[0].click()
            
            # Should show details modal
            modal = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "shortage-modal"))
            )
            
            # Click create PO button
            create_po_btn = modal.find_element(By.CLASS_NAME, "create-po-btn")
            create_po_btn.click()
            
            # Verify PO was created
            success_msg = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
            )
            
            assert "Purchase order created" in success_msg.text
    
    def test_production_planning_e2e(self, api_client):
        """Test production planning end-to-end via API"""
        
        # 1. Get pending production orders
        orders = api_client.get('/api/v2/production/orders?status=pending').json()
        
        assert len(orders) > 0, "No pending orders to test"
        
        # 2. Check yarn availability for first order
        order = orders[0]
        yarn_check = api_client.post('/api/v2/yarn/availability-check',
                                    json={'style_id': order['style_id'],
                                          'quantity': order['quantity']}).json()
        
        # 3. If yarns available, assign to machine
        if yarn_check['available']:
            assignment = api_client.post('/api/v2/production/assign-machine',
                                        json={'order_id': order['order_id']}).json()
            
            assert assignment['status'] == 'success'
            assert 'machine_id' in assignment
            
            # 4. Verify order status updated
            updated_order = api_client.get(f'/api/v2/production/orders/{order["order_id"]}').json()
            assert updated_order['status'] == 'assigned'
            assert updated_order['machine_id'] == assignment['machine_id']
```

---

## Phase 6: Infrastructure & Deployment (Week 7)

### Objective
Establish scalable infrastructure and automated deployment pipeline.

### Day 1: Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY web/ ./web/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5006

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5006/health')"

# Run application
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5006", "--timeout=120", "src.app:create_app()"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  erp-api:
    build: .
    ports:
      - "5006:5006"
    environment:
      - REDIS_HOST=redis
      - DB_HOST=postgres
      - ENV=production
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: beverly_knits
      POSTGRES_USER: erp_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - erp-api

volumes:
  redis_data:
  postgres_data:
```

### Day 2: Kubernetes Deployment
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: beverly-knits

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: erp-api
  namespace: beverly-knits
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: erp-api
  template:
    metadata:
      labels:
        app: erp-api
    spec:
      containers:
      - name: erp-api
        image: beverly-knits/erp:v2.0.0
        ports:
        - containerPort: 5006
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: DB_HOST
          value: postgres-service
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5006
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5006
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: erp-api-service
  namespace: beverly-knits
spec:
  type: LoadBalancer
  selector:
    app: erp-api
  ports:
  - port: 80
    targetPort: 5006
    protocol: TCP

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: erp-api-hpa
  namespace: beverly-knits
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: erp-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Day 3: CI/CD Pipeline
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: beverly-knits/erp

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
    
    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        mypy src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=sha
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/erp-api \
          erp-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n beverly-knits
        
        kubectl rollout status deployment/erp-api -n beverly-knits
    
    - name: Run smoke tests
      run: |
        ./scripts/smoke-tests.sh
```

### Day 4: Monitoring Setup
```python
# src/monitoring/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Response
import time

# Define metrics
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_latency = Histogram(
    'api_latency_seconds',
    'API latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

yarn_shortage_gauge = Gauge(
    'yarn_shortages',
    'Current number of yarn shortages',
    ['severity']
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio'
)

def setup_metrics(app):
    """Setup Prometheus metrics for Flask app"""
    
    @app.before_request
    def before_request():
        request.start_time = time.time()
        active_connections.inc()
    
    @app.after_request
    def after_request(response):
        # Record metrics
        latency = time.time() - request.start_time
        api_latency.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown'
        ).observe(latency)
        
        api_requests.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
        
        active_connections.dec()
        
        return response
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        # Update business metrics
        update_business_metrics()
        
        return Response(generate_latest(), mimetype='text/plain')

def update_business_metrics():
    """Update business-specific metrics"""
    
    # Update yarn shortage metrics
    shortages = services.get('inventory').get_shortages()
    yarn_shortage_gauge.labels(severity='critical').set(
        len(shortages[shortages['planning_balance'] < -1000])
    )
    yarn_shortage_gauge.labels(severity='warning').set(
        len(shortages[(shortages['planning_balance'] >= -1000) & 
                     (shortages['planning_balance'] < 0)])
    )
    
    # Update cache metrics
    cache_stats = services.get('cache').get_stats()
    if cache_stats['requests'] > 0:
        hit_ratio = cache_stats['hits'] / cache_stats['requests']
        cache_hit_ratio.set(hit_ratio)
```

### Day 5: Load Balancing & Auto-scaling
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: erp-ingress
  namespace: beverly-knits
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - erp.beverly-knits.com
    secretName: erp-tls-secret
  rules:
  - host: erp.beverly-knits.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: erp-api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: erp-config
  namespace: beverly-knits
data:
  app.conf: |
    # Application configuration
    DEBUG=false
    LOG_LEVEL=INFO
    MAX_WORKERS=4
    CACHE_TTL=300
    DATABASE_POOL_SIZE=20
    
  feature-flags.json: |
    {
      "use_new_inventory_service": true,
      "enable_ml_forecasting": true,
      "enable_websocket_updates": true,
      "cache_warming_enabled": true
    }
```

---

## Phase 7: Documentation & Handoff (Week 8)

### Objective
Complete documentation and ensure smooth knowledge transfer.

### Day 1-2: API Documentation
```python
# src/api/openapi_spec.py
"""
Generate OpenAPI specification for the API
"""

openapi_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Beverly Knits ERP API",
        "version": "2.0.0",
        "description": "Production-ready textile manufacturing ERP API"
    },
    "servers": [
        {"url": "https://erp.beverly-knits.com/api/v2", "description": "Production"},
        {"url": "http://localhost:5006/api/v2", "description": "Development"}
    ],
    "paths": {
        "/inventory": {
            "get": {
                "summary": "Get inventory intelligence",
                "parameters": [
                    {
                        "name": "view",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["summary", "detailed", "shortages"]}
                    },
                    {
                        "name": "realtime",
                        "in": "query",
                        "schema": {"type": "boolean"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Inventory data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/InventoryResponse"}
                            }
                        }
                    }
                }
            }
        }
        # ... more endpoints
    },
    "components": {
        "schemas": {
            "InventoryResponse": {
                "type": "object",
                "properties": {
                    "total_items": {"type": "integer"},
                    "shortage_count": {"type": "integer"},
                    "critical_shortages": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/YarnShortage"}
                    }
                }
            }
        }
    }
}
```

### Day 3-5: Complete Documentation Package

Create comprehensive documentation:
- Architecture diagrams
- Deployment guide
- Operations manual
- Troubleshooting guide
- Development guide

---

## Technical Details

### Required Dependencies
```python
# requirements.txt
flask==3.0.0
pandas==2.0.3
numpy==1.24.3
redis==5.0.0
sqlalchemy==2.0.0
gunicorn==21.2.0
pytest==7.4.0
prometheus-client==0.17.1
marshmallow==3.20.0
alembic==1.12.0
```

### Configuration Files
```python
# src/config/settings.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Application
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://localhost/beverly_knits')
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '20'))
    
    # Redis
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    
    # Performance
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '4'))
```

---

## Risk Management

### Technical Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Service extraction breaks functionality | Medium | High | Comprehensive testing, feature flags |
| Performance regression | Low | High | Continuous benchmarking |
| Data inconsistency | Low | Critical | Validation, parallel runs |
| Integration failures | Medium | Medium | Gradual rollout, rollback plan |

### Rollback Strategy
```bash
#!/bin/bash
# rollback.sh

# Check system health
ERROR_RATE=$(curl -s http://localhost:5006/metrics | grep error_rate)

if [ "$ERROR_RATE" -gt "5" ]; then
    echo "High error rate detected: $ERROR_RATE%"
    
    # Revert to previous version
    kubectl rollout undo deployment/erp-api -n beverly-knits
    
    # Disable feature flags
    kubectl patch configmap erp-config -n beverly-knits \
        --type merge -p '{"data":{"feature-flags.json":"{\"use_new_inventory_service\":false}"}}'
    
    # Alert team
    curl -X POST $SLACK_WEBHOOK -d '{"text":"Automatic rollback triggered"}'
fi
```

---

## Success Metrics

### Technical KPIs
| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Monolith size | 18,076 lines | <2,000 lines | Line count |
| API response time | Variable | <200ms p95 | Prometheus |
| Test coverage | Unknown | 80%+ | pytest-cov |
| Memory usage | Unoptimized | -50% | Monitoring |
| Cache hit rate | 70-90% | >90% | Redis stats |

### Business KPIs
| Metric | Current | Target | Impact |
|--------|---------|--------|--------|
| Feature completion | 90% | 100% | User satisfaction |
| Bug rate | Unknown | <5/week | Quality |
| Deploy frequency | Manual | Daily | Agility |
| System uptime | Unknown | 99.9% | Reliability |

---

## Immediate Next Steps

### Week 1 - Start Immediately
1. **Day 1**: Extract ProductionSchedulerService from monolith
2. **Day 2**: Extract ManufacturingSupplyChainService
3. **Day 3**: Wire up existing extracted services
4. **Day 4**: Start migrating API endpoints to v2
5. **Day 5**: Complete service registry implementation

### Success Criteria for Week 1
- [ ] All services extracted from monolith
- [ ] At least 50% of API endpoints migrated
- [ ] Services wired up and working
- [ ] No regression in functionality
- [ ] All tests passing

### Team Assignments
- **Developer 1**: Service extraction (Days 1-2)
- **Developer 2**: API migration (Days 3-5)  
- **Developer 3**: Testing & validation (Continuous)
- **DevOps**: Environment setup (Day 1)

---

## Conclusion

This comprehensive implementation plan provides a clear path to transform Beverly Knits ERP v2 from a monolithic application to a modern, scalable microservices architecture. By focusing on architectural improvements and performance optimization first (while deferring authentication), we can deliver immediate business value through:

1. **10-100x performance improvements** from vectorization
2. **90% reduction in codebase complexity**
3. **100% feature completion** including missing APIs
4. **Scalable architecture** ready for growth

The 8-week timeline is aggressive but achievable with dedicated resources and clear priorities. Each phase builds upon the previous one, ensuring continuous progress while maintaining system stability.

**Ready to begin implementation immediately.**