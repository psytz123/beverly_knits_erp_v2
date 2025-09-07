#!/usr/bin/env python3
"""
eFab Automatic Data Synchronization Service
Schedules and manages periodic sync of eFab data to the ERP system
"""

import time
import threading
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2')

from src.data_sync.efab_api_connector import eFabAPIConnector
from src.utils.cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class eFabAutoSync:
    """Automatic synchronization service for eFab data"""
    
    def __init__(self, 
                 sync_interval_minutes: int = 15,
                 enable_auto_sync: bool = True,
                 config_path: Optional[str] = None):
        """
        Initialize the auto-sync service
        
        Args:
            sync_interval_minutes: Minutes between sync operations
            enable_auto_sync: Whether to enable automatic synchronization
            config_path: Path to configuration file
        """
        self.sync_interval = sync_interval_minutes
        self.enabled = enable_auto_sync
        self.config_path = config_path or "/mnt/c/finalee/beverly_knits_erp_v2/config/efab_config.json"
        self.connector = None
        self.cache_manager = CacheManager()
        self.last_sync = None
        self.sync_thread = None
        self.running = False
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'last_error': None,
            'total_records_synced': 0
        }
        
        # Load configuration
        self._load_config()
        
        # Initialize connector
        self._init_connector()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.sync_interval = config.get('sync_interval_minutes', self.sync_interval)
                    self.enabled = config.get('auto_sync_enabled', self.enabled)
                    logger.info(f"Configuration loaded: sync every {self.sync_interval} minutes")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def _init_connector(self):
        """Initialize the eFab API connector"""
        try:
            self.connector = eFabAPIConnector()
            logger.info("eFab connector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connector: {e}")
            self.enabled = False
    
    def _clear_cache_safely(self, pattern: str):
        """Safe cache clearing that handles different cache manager versions"""
        try:
            if hasattr(self.cache_manager, 'clear_pattern'):
                self.cache_manager.clear_pattern(pattern)
            elif hasattr(self.cache_manager, 'invalidate_pattern'):
                self.cache_manager.invalidate_pattern(pattern)
            else:
                logger.debug(f"Cache manager doesn't support pattern clearing: {pattern}")
        except Exception as e:
            logger.debug(f"Cache clear failed for pattern {pattern}: {e}")
    
    def sync_data(self) -> Dict[str, Any]:
        """
        Perform a single sync operation
        
        Returns:
            Dictionary with sync results
        """
        logger.info("Starting eFab data synchronization...")
        self.sync_stats['total_syncs'] += 1
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'data_synced': {},
            'errors': []
        }
        
        try:
            if not self.connector:
                self._init_connector()
                if not self.connector:
                    raise Exception("Connector not available")
            
            # Test connection first
            if not self.connector.test_connection():
                raise Exception("Connection test failed")
            
            records_synced = 0
            
            # Sync sales orders
            try:
                sales_orders = self.connector.get_sales_order_plan_list()
                if sales_orders is not None and not sales_orders.empty:
                    path = self.connector.sync_to_csv(sales_orders, 'eFab_SO_List.csv')
                    results['data_synced']['sales_orders'] = {
                        'count': len(sales_orders),
                        'path': path
                    }
                    records_synced += len(sales_orders)
                    
                    # Clear related cache
                    self._clear_cache_safely('*sales*')
                    self._clear_cache_safely('*order*')
            except Exception as e:
                logger.error(f"Error syncing sales orders: {e}")
                results['errors'].append(f"Sales orders: {str(e)}")
            
            # Sync knit orders
            try:
                knit_orders = self.connector.get_knit_orders()
                if knit_orders is not None and not knit_orders.empty:
                    path = self.connector.sync_to_csv(knit_orders, 'eFab_Knit_Orders.csv')
                    results['data_synced']['knit_orders'] = {
                        'count': len(knit_orders),
                        'path': path
                    }
                    records_synced += len(knit_orders)
                    
                    # Clear related cache
                    self._clear_cache_safely('*knit*')
                    self._clear_cache_safely('*production*')
            except Exception as e:
                logger.error(f"Error syncing knit orders: {e}")
                results['errors'].append(f"Knit orders: {str(e)}")
            
            # Sync inventory for all warehouses
            for warehouse in ['F01', 'G00', 'G02', 'I01']:
                try:
                    inventory = self.connector.get_inventory_data(warehouse)
                    if inventory is not None and not inventory.empty:
                        path = self.connector.sync_to_csv(inventory, f'eFab_Inventory_{warehouse}.csv')
                        results['data_synced'][f'inventory_{warehouse}'] = {
                            'count': len(inventory),
                            'path': path
                        }
                        records_synced += len(inventory)
                except Exception as e:
                    logger.error(f"Error syncing inventory for {warehouse}: {e}")
                    results['errors'].append(f"Inventory {warehouse}: {str(e)}")
            
            # Clear inventory cache
            self._clear_cache_safely('*inventory*')
            self._clear_cache_safely('*stock*')
            
            # Update statistics
            self.sync_stats['successful_syncs'] += 1
            self.sync_stats['total_records_synced'] += records_synced
            self.last_sync = datetime.now()
            
            results['success'] = True
            results['records_synced'] = records_synced
            
            logger.info(f"Sync completed successfully. {records_synced} records synced.")
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.sync_stats['failed_syncs'] += 1
            self.sync_stats['last_error'] = str(e)
            results['errors'].append(str(e))
        
        return results
    
    def _sync_job(self):
        """Job function for scheduled sync"""
        try:
            self.sync_data()
        except Exception as e:
            logger.error(f"Scheduled sync failed: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        schedule.every(self.sync_interval).minutes.do(self._sync_job)
        
        # Run initial sync
        self._sync_job()
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the automatic synchronization service"""
        if not self.enabled:
            logger.warning("Auto-sync is disabled")
            return False
        
        if self.running:
            logger.warning("Auto-sync is already running")
            return False
        
        logger.info(f"Starting auto-sync service (interval: {self.sync_interval} minutes)")
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.sync_thread.start()
        
        return True
    
    def stop(self):
        """Stop the automatic synchronization service"""
        logger.info("Stopping auto-sync service...")
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        
        logger.info("Auto-sync service stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sync service"""
        return {
            'enabled': self.enabled,
            'running': self.running,
            'sync_interval_minutes': self.sync_interval,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'next_sync': self._get_next_sync_time(),
            'statistics': self.sync_stats
        }
    
    def _get_next_sync_time(self) -> Optional[str]:
        """Calculate the next sync time"""
        if not self.running or not self.last_sync:
            return None
        
        next_sync = self.last_sync + timedelta(minutes=self.sync_interval)
        return next_sync.isoformat()
    
    def force_sync(self) -> Dict[str, Any]:
        """Force an immediate sync operation"""
        logger.info("Forcing immediate sync...")
        return self.sync_data()


# Global instance for the service
_auto_sync_service = None


def get_auto_sync_service() -> eFabAutoSync:
    """Get or create the global auto-sync service instance"""
    global _auto_sync_service
    if _auto_sync_service is None:
        _auto_sync_service = eFabAutoSync()
    return _auto_sync_service


def start_auto_sync(interval_minutes: int = 15) -> bool:
    """Start the auto-sync service"""
    service = get_auto_sync_service()
    service.sync_interval = interval_minutes
    return service.start()


def stop_auto_sync():
    """Stop the auto-sync service"""
    service = get_auto_sync_service()
    service.stop()


def get_sync_status() -> Dict[str, Any]:
    """Get the status of the auto-sync service"""
    service = get_auto_sync_service()
    return service.get_status()


def force_sync_now() -> Dict[str, Any]:
    """Force an immediate sync"""
    service = get_auto_sync_service()
    return service.force_sync()


if __name__ == "__main__":
    # Test the auto-sync service
    import argparse
    
    parser = argparse.ArgumentParser(description='eFab Auto-Sync Service')
    parser.add_argument('--interval', type=int, default=15, help='Sync interval in minutes')
    parser.add_argument('--once', action='store_true', help='Run sync once and exit')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon service')
    
    args = parser.parse_args()
    
    if args.once:
        # Run sync once
        service = eFabAutoSync()
        result = service.sync_data()
        print(json.dumps(result, indent=2))
    elif args.daemon:
        # Run as daemon
        service = eFabAutoSync(sync_interval_minutes=args.interval)
        
        try:
            service.start()
            print(f"Auto-sync service started (interval: {args.interval} minutes)")
            print("Press Ctrl+C to stop...")
            
            while True:
                time.sleep(60)
                status = service.get_status()
                print(f"Status: {status['statistics']['successful_syncs']} successful, "
                      f"{status['statistics']['failed_syncs']} failed")
        except KeyboardInterrupt:
            print("\nStopping service...")
            service.stop()
    else:
        # Run interactively
        service = eFabAutoSync(sync_interval_minutes=args.interval)
        service.start()
        
        print(f"\n✨ eFab Auto-Sync Service Started")
        print(f"Sync interval: {args.interval} minutes")
        print("\nCommands:")
        print("  status - Show sync status")
        print("  sync   - Force sync now")
        print("  stop   - Stop service")
        print("  quit   - Exit\n")
        
        while True:
            try:
                cmd = input("> ").strip().lower()
                
                if cmd == 'status':
                    status = service.get_status()
                    print(json.dumps(status, indent=2))
                elif cmd == 'sync':
                    result = service.force_sync()
                    print(f"Sync {'successful' if result['success'] else 'failed'}")
                    print(f"Records synced: {result.get('records_synced', 0)}")
                elif cmd == 'stop':
                    service.stop()
                    print("Service stopped")
                elif cmd in ['quit', 'exit']:
                    service.stop()
                    break
                else:
                    print("Unknown command")
            except KeyboardInterrupt:
                service.stop()
                break
        
        print("\nGoodbye!")