"""Enhanced cache warming implementation for critical data."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.cache.multi_tier_cache import MultiTierCache
from src.infrastructure.data.unified_data_loader import UnifiedDataLoader


class CacheWarmingStrategy:
    """Strategy for cache warming configuration."""
    
    def __init__(
        self,
        data_types: List[str],
        warm_on_startup: bool = True,
        warm_interval_minutes: int = 30,
        priority: int = 1,
        parallel: bool = True
    ):
        """
        Initialize cache warming strategy.
        
        Args:
            data_types: List of data types to warm
            warm_on_startup: Whether to warm on application startup
            warm_interval_minutes: Interval for periodic warming
            priority: Priority level (1=highest)
            parallel: Whether to warm in parallel
        """
        self.data_types = data_types
        self.warm_on_startup = warm_on_startup
        self.warm_interval_minutes = warm_interval_minutes
        self.priority = priority
        self.parallel = parallel
        self.last_warmed = {}


class EnhancedCacheWarmer:
    """
    Enhanced cache warmer with strategies and monitoring.
    Implements proactive cache warming for critical data.
    """
    
    def __init__(
        self,
        cache: MultiTierCache,
        data_loader: UnifiedDataLoader,
        max_workers: int = 4
    ):
        """Initialize enhanced cache warmer."""
        self.cache = cache
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Define warming strategies
        self.strategies = {
            'critical': CacheWarmingStrategy(
                data_types=['yarn_inventory', 'bom_data', 'production_orders'],
                warm_on_startup=True,
                warm_interval_minutes=15,
                priority=1,
                parallel=True
            ),
            'important': CacheWarmingStrategy(
                data_types=['work_centers', 'machine_report'],
                warm_on_startup=True,
                warm_interval_minutes=30,
                priority=2,
                parallel=True
            ),
            'standard': CacheWarmingStrategy(
                data_types=['sales_activity', 'demand_data'],
                warm_on_startup=False,
                warm_interval_minutes=60,
                priority=3,
                parallel=False
            )
        }
        
        # Warming statistics
        self.stats = {
            'total_warmings': 0,
            'successful_warmings': 0,
            'failed_warmings': 0,
            'last_warming_time': None,
            'total_time_ms': 0,
            'data_type_stats': {}
        }
        
        # Background tasks
        self.warming_tasks = []
        self.is_running = False
    
    async def warm_on_startup(self):
        """
        Warm critical data on application startup.
        Executes warming strategies marked for startup.
        """
        self.logger.info("Starting cache warming on startup...")
        start_time = datetime.now()
        
        # Sort strategies by priority
        sorted_strategies = sorted(
            self.strategies.items(),
            key=lambda x: x[1].priority
        )
        
        results = {}
        
        for strategy_name, strategy in sorted_strategies:
            if not strategy.warm_on_startup:
                continue
            
            self.logger.info(f"Warming {strategy_name} data (priority {strategy.priority})")
            
            if strategy.parallel:
                # Warm in parallel
                strategy_results = await self._warm_parallel(
                    strategy.data_types,
                    strategy_name
                )
            else:
                # Warm sequentially
                strategy_results = await self._warm_sequential(
                    strategy.data_types,
                    strategy_name
                )
            
            results[strategy_name] = strategy_results
            
            # Update last warmed time
            for data_type in strategy.data_types:
                strategy.last_warmed[data_type] = datetime.now()
        
        # Calculate statistics
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.stats['total_warmings'] += 1
        self.stats['last_warming_time'] = datetime.now()
        self.stats['total_time_ms'] += elapsed_ms
        
        # Count successes and failures
        total_success = sum(
            1 for strategy_results in results.values()
            for result in strategy_results.values()
            if result['success']
        )
        total_failure = sum(
            1 for strategy_results in results.values()
            for result in strategy_results.values()
            if not result['success']
        )
        
        self.stats['successful_warmings'] += total_success
        self.stats['failed_warmings'] += total_failure
        
        self.logger.info(
            f"Cache warming completed in {elapsed_ms:.2f}ms. "
            f"Success: {total_success}, Failed: {total_failure}"
        )
        
        return results
    
    async def _warm_parallel(
        self,
        data_types: List[str],
        strategy_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Warm multiple data types in parallel."""
        tasks = []
        
        for data_type in data_types:
            task = self._warm_single_data_type(data_type, strategy_name)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        warm_results = {}
        for data_type, result in zip(data_types, results):
            if isinstance(result, Exception):
                warm_results[data_type] = {
                    'success': False,
                    'error': str(result),
                    'timestamp': datetime.now()
                }
            else:
                warm_results[data_type] = result
        
        return warm_results
    
    async def _warm_sequential(
        self,
        data_types: List[str],
        strategy_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Warm multiple data types sequentially."""
        warm_results = {}
        
        for data_type in data_types:
            try:
                result = await self._warm_single_data_type(data_type, strategy_name)
                warm_results[data_type] = result
            except Exception as e:
                warm_results[data_type] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        return warm_results
    
    async def _warm_single_data_type(
        self,
        data_type: str,
        strategy_name: str
    ) -> Dict[str, Any]:
        """Warm a single data type."""
        start_time = datetime.now()
        
        try:
            # Load data using the data loader
            if data_type == 'yarn_inventory':
                data = await self.data_loader.load_yarn_inventory()
            elif data_type == 'bom_data':
                data = await self.data_loader.load_bom_data()
            elif data_type == 'production_orders':
                data = await self.data_loader.load_production_orders()
            elif data_type == 'work_centers':
                data = await self.data_loader.load_work_centers()
            elif data_type == 'machine_report':
                data = await self.data_loader.load_machine_report()
            elif data_type == 'sales_activity':
                data = await self.data_loader.load_sales_activity()
            elif data_type == 'demand_data':
                data = await self.data_loader.load_demand_data()
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Data is automatically cached by the data loader
            rows = len(data) if isinstance(data, pd.DataFrame) else 0
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            if data_type not in self.stats['data_type_stats']:
                self.stats['data_type_stats'][data_type] = {
                    'warm_count': 0,
                    'total_time_ms': 0,
                    'avg_time_ms': 0,
                    'last_row_count': 0
                }
            
            stats = self.stats['data_type_stats'][data_type]
            stats['warm_count'] += 1
            stats['total_time_ms'] += elapsed_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['warm_count']
            stats['last_row_count'] = rows
            
            self.logger.debug(
                f"Warmed {data_type} ({rows} rows) in {elapsed_ms:.2f}ms "
                f"[{strategy_name}]"
            )
            
            return {
                'success': True,
                'rows': rows,
                'elapsed_ms': elapsed_ms,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to warm {data_type}: {e}")
            
            # Update failure statistics
            if data_type not in self.stats['data_type_stats']:
                self.stats['data_type_stats'][data_type] = {'failures': 0}
            else:
                if 'failures' not in self.stats['data_type_stats'][data_type]:
                    self.stats['data_type_stats'][data_type]['failures'] = 0
            
            self.stats['data_type_stats'][data_type]['failures'] += 1
            
            raise
    
    async def start_periodic_warming(self):
        """Start periodic cache warming based on strategies."""
        if self.is_running:
            self.logger.warning("Periodic warming already running")
            return
        
        self.is_running = True
        self.logger.info("Starting periodic cache warming")
        
        # Create tasks for each strategy
        for strategy_name, strategy in self.strategies.items():
            task = asyncio.create_task(
                self._periodic_warming_loop(strategy_name, strategy)
            )
            self.warming_tasks.append(task)
    
    async def _periodic_warming_loop(
        self,
        strategy_name: str,
        strategy: CacheWarmingStrategy
    ):
        """Periodic warming loop for a strategy."""
        while self.is_running:
            try:
                # Wait for the warming interval
                await asyncio.sleep(strategy.warm_interval_minutes * 60)
                
                # Check if warming is needed
                needs_warming = []
                for data_type in strategy.data_types:
                    last_warmed = strategy.last_warmed.get(data_type)
                    if not last_warmed or \
                       (datetime.now() - last_warmed).total_seconds() > strategy.warm_interval_minutes * 60:
                        needs_warming.append(data_type)
                
                if needs_warming:
                    self.logger.info(
                        f"Periodic warming for {strategy_name}: {needs_warming}"
                    )
                    
                    if strategy.parallel:
                        await self._warm_parallel(needs_warming, strategy_name)
                    else:
                        await self._warm_sequential(needs_warming, strategy_name)
                    
                    # Update last warmed times
                    for data_type in needs_warming:
                        strategy.last_warmed[data_type] = datetime.now()
                        
            except Exception as e:
                self.logger.error(f"Error in periodic warming loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def stop_periodic_warming(self):
        """Stop periodic cache warming."""
        self.is_running = False
        
        # Cancel all warming tasks
        for task in self.warming_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.warming_tasks, return_exceptions=True)
        
        self.warming_tasks = []
        self.logger.info("Periodic cache warming stopped")
    
    async def warm_specific(self, data_types: List[str]):
        """Manually warm specific data types."""
        self.logger.info(f"Manual cache warming for: {data_types}")
        
        results = await self._warm_parallel(data_types, 'manual')
        
        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        
        self.logger.info(
            f"Manual warming completed. Success: {successful}, Failed: {failed}"
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache warming statistics."""
        return {
            'overall': {
                'total_warmings': self.stats['total_warmings'],
                'successful_warmings': self.stats['successful_warmings'],
                'failed_warmings': self.stats['failed_warmings'],
                'success_rate': (
                    self.stats['successful_warmings'] / 
                    max(self.stats['total_warmings'], 1) * 100
                ),
                'last_warming_time': self.stats['last_warming_time'].isoformat()
                    if self.stats['last_warming_time'] else None,
                'avg_warming_time_ms': (
                    self.stats['total_time_ms'] / max(self.stats['total_warmings'], 1)
                )
            },
            'by_data_type': self.stats['data_type_stats'],
            'strategies': {
                name: {
                    'data_types': strategy.data_types,
                    'warm_on_startup': strategy.warm_on_startup,
                    'interval_minutes': strategy.warm_interval_minutes,
                    'priority': strategy.priority,
                    'last_warmed': {
                        dt: lw.isoformat() if lw else None
                        for dt, lw in strategy.last_warmed.items()
                    }
                }
                for name, strategy in self.strategies.items()
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)