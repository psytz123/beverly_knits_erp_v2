#!/usr/bin/env python3
"""
eFab Integration Blueprint for Beverly Knits ERP
Provides API endpoints to integrate eFab data into the ERP system
"""

from flask import Blueprint, jsonify, request
import pandas as pd
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2')

from src.data_sync.efab_api_connector import eFabAPIConnector
from src.utils.cache_manager import CacheManager
from src.data_sync.efab_auto_sync import (
    get_auto_sync_service,
    start_auto_sync,
    stop_auto_sync,
    get_sync_status,
    force_sync_now
)

logger = logging.getLogger(__name__)

# Create blueprint
efab_bp = Blueprint('efab', __name__, url_prefix='/api/efab')

# Initialize cache manager
cache_manager = CacheManager()

# Start auto-sync service on blueprint registration
try:
    start_auto_sync(interval_minutes=15)
    logger.info("eFab auto-sync service started")
except Exception as e:
    logger.warning(f"Could not start auto-sync service: {e}")


@efab_bp.route('/status', methods=['GET'])
def get_connection_status():
    """
    Check eFab API connection status
    """
    try:
        connector = eFabAPIConnector()
        is_connected = connector.test_connection()
        
        return jsonify({
            'success': True,
            'connected': is_connected,
            'message': 'Connected to eFab API' if is_connected else 'Not connected to eFab API',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error checking connection status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/sync', methods=['POST'])
def sync_data():
    """
    Sync all data from eFab API
    """
    try:
        # Get options from request
        data_types = request.json.get('data_types', ['all']) if request.json else ['all']
        force_refresh = request.json.get('force_refresh', False) if request.json else False
        
        # Clear cache if force refresh
        if force_refresh:
            cache_manager.clear_all_caches()
        
        connector = eFabAPIConnector()
        results = {}
        
        # Sync requested data types
        if 'all' in data_types or 'sales_orders' in data_types:
            sales_orders = connector.get_sales_order_plan_list()
            if sales_orders is not None:
                path = connector.sync_to_csv(sales_orders, 'eFab_SO_List.csv')
                results['sales_orders'] = {
                    'records': len(sales_orders),
                    'path': path
                }
        
        if 'all' in data_types or 'knit_orders' in data_types:
            knit_orders = connector.get_knit_orders()
            if knit_orders is not None:
                path = connector.sync_to_csv(knit_orders, 'eFab_Knit_Orders.csv')
                results['knit_orders'] = {
                    'records': len(knit_orders),
                    'path': path
                }
        
        if 'all' in data_types or 'inventory' in data_types:
            for warehouse in ['F01', 'G00', 'G02', 'I01']:
                inventory = connector.get_inventory_data(warehouse)
                if inventory is not None:
                    path = connector.sync_to_csv(inventory, f'eFab_Inventory_{warehouse}.csv')
                    results[f'inventory_{warehouse}'] = {
                        'records': len(inventory),
                        'path': path
                    }
        
        return jsonify({
            'success': True,
            'synced_data': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error syncing data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/sales-orders', methods=['GET'])
def get_sales_orders():
    """
    Get sales order data from eFab
    """
    try:
        # Check cache first
        cache_key = 'efab_sales_orders'
        cached_data = cache_manager.get(cache_key)
        
        if cached_data and not request.args.get('force_refresh'):
            return jsonify({
                'success': True,
                'data': cached_data,
                'source': 'cache'
            })
        
        # Fetch from API
        connector = eFabAPIConnector()
        sales_orders = connector.get_sales_order_plan_list()
        
        if sales_orders is not None:
            data = sales_orders.to_dict(orient='records')
            
            # Cache the data
            cache_manager.set(cache_key, data, ttl=900)  # 15 minutes
            
            return jsonify({
                'success': True,
                'data': data,
                'count': len(data),
                'source': 'api'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not fetch sales orders'
            }), 404
            
    except Exception as e:
        logger.error(f"Error fetching sales orders: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/knit-orders', methods=['GET'])
def get_knit_orders():
    """
    Get knit order data from eFab
    """
    try:
        # Check cache first
        cache_key = 'efab_knit_orders'
        cached_data = cache_manager.get(cache_key)
        
        if cached_data and not request.args.get('force_refresh'):
            return jsonify({
                'success': True,
                'data': cached_data,
                'source': 'cache'
            })
        
        # Fetch from API
        connector = eFabAPIConnector()
        knit_orders = connector.get_knit_orders()
        
        if knit_orders is not None:
            data = knit_orders.to_dict(orient='records')
            
            # Cache the data
            cache_manager.set(cache_key, data, ttl=900)  # 15 minutes
            
            return jsonify({
                'success': True,
                'data': data,
                'count': len(data),
                'source': 'api'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not fetch knit orders'
            }), 404
            
    except Exception as e:
        logger.error(f"Error fetching knit orders: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/inventory/<warehouse>', methods=['GET'])
def get_inventory(warehouse: str):
    """
    Get inventory data for specific warehouse from eFab
    """
    try:
        # Validate warehouse
        valid_warehouses = ['F01', 'G00', 'G02', 'I01', 'all']
        if warehouse not in valid_warehouses:
            return jsonify({
                'success': False,
                'error': f'Invalid warehouse. Must be one of: {valid_warehouses}'
            }), 400
        
        # Check cache first
        cache_key = f'efab_inventory_{warehouse}'
        cached_data = cache_manager.get(cache_key)
        
        if cached_data and not request.args.get('force_refresh'):
            return jsonify({
                'success': True,
                'data': cached_data,
                'warehouse': warehouse,
                'source': 'cache'
            })
        
        # Fetch from API
        connector = eFabAPIConnector()
        inventory = connector.get_inventory_data(warehouse)
        
        if inventory is not None:
            data = inventory.to_dict(orient='records')
            
            # Cache the data
            cache_manager.set(cache_key, data, ttl=900)  # 15 minutes
            
            return jsonify({
                'success': True,
                'data': data,
                'count': len(data),
                'warehouse': warehouse,
                'source': 'api'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not fetch inventory for {warehouse}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error fetching inventory: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/auto-sync/status', methods=['GET'])
def get_auto_sync_status():
    """
    Get the status of the auto-sync service
    """
    try:
        status = get_sync_status()
        return jsonify({
            'success': True,
            **status
        })
    except Exception as e:
        logger.error(f"Error getting auto-sync status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/auto-sync/start', methods=['POST'])
def start_auto_sync_endpoint():
    """
    Start the auto-sync service
    """
    try:
        interval = request.json.get('interval_minutes', 15) if request.json else 15
        success = start_auto_sync(interval_minutes=interval)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Auto-sync started with {interval} minute interval'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Auto-sync already running or disabled'
            }), 400
            
    except Exception as e:
        logger.error(f"Error starting auto-sync: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/auto-sync/stop', methods=['POST'])
def stop_auto_sync_endpoint():
    """
    Stop the auto-sync service
    """
    try:
        stop_auto_sync()
        return jsonify({
            'success': True,
            'message': 'Auto-sync stopped'
        })
    except Exception as e:
        logger.error(f"Error stopping auto-sync: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/auto-sync/force', methods=['POST'])
def force_sync_endpoint():
    """
    Force an immediate sync
    """
    try:
        result = force_sync_now()
        return jsonify({
            'success': result['success'],
            'result': result
        })
    except Exception as e:
        logger.error(f"Error forcing sync: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@efab_bp.route('/config', methods=['GET', 'POST'])
def manage_config():
    """
    Get or update eFab API configuration
    """
    config_path = '/mnt/c/finalee/beverly_knits_erp_v2/config/efab_config.json'
    
    try:
        if request.method == 'GET':
            # Read current config
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Don't expose the full session cookie
                if 'session_cookie' in config:
                    config['session_cookie'] = config['session_cookie'][:10] + '...'
                
                return jsonify({
                    'success': True,
                    'config': config
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Configuration file not found'
                }), 404
        
        elif request.method == 'POST':
            # Update config
            new_config = request.json
            
            if not new_config:
                return jsonify({
                    'success': False,
                    'error': 'No configuration data provided'
                }), 400
            
            # Read existing config
            import json
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update only allowed fields
            allowed_fields = ['session_cookie', 'sync_interval_minutes', 'retry_attempts', 'timeout_seconds']
            for field in allowed_fields:
                if field in new_config:
                    config[field] = new_config[field]
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully'
            })
            
    except Exception as e:
        logger.error(f"Error managing config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500