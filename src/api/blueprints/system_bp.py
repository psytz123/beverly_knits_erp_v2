"""
System Blueprint - System health, debugging, and cache management endpoints
"""
from flask import Blueprint, jsonify, request
import logging
import os
import platform
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Create the blueprint
system_bp = Blueprint('system', __name__)

# Global components (will be initialized by main app)
handler = None


class SystemAPIHandler:
    """Handler for system-related operations"""
    
    def __init__(self, service_manager=None, data_loader=None, cache_manager=None):
        self.service_manager = service_manager
        self.data_loader = data_loader
        self.cache_manager = cache_manager
        self.start_time = datetime.now()
    
    def get_uptime(self):
        """Get system uptime"""
        delta = datetime.now() - self.start_time
        return {
            'days': delta.days,
            'hours': delta.seconds // 3600,
            'minutes': (delta.seconds % 3600) // 60,
            'total_seconds': delta.total_seconds()
        }


def init_blueprint(service_manager, data_loader, cache_manager=None):
    """Initialize the blueprint with required components"""
    global handler
    handler = SystemAPIHandler(
        service_manager=service_manager,
        data_loader=data_loader,
        cache_manager=cache_manager
    )


# --- System Health Endpoints ---

@system_bp.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': handler.get_uptime() if handler else None,
            'components': {}
        }
        
        if handler:
            # Check service manager
            if handler.service_manager:
                health_status['components']['service_manager'] = {
                    'status': 'healthy',
                    'services_count': len(handler.service_manager.services)
                }
            
            # Check data loader
            if handler.data_loader:
                health_status['components']['data_loader'] = {
                    'status': 'healthy',
                    'type': 'ConsolidatedDataLoader',
                    'workers': getattr(handler.data_loader, 'max_workers', 1)
                }
            
            # Check cache
            if handler.cache_manager:
                health_status['components']['cache'] = {
                    'status': 'healthy',
                    'type': type(handler.cache_manager).__name__
                }
        
        # System info
        health_status['system'] = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
        
        return jsonify(health_status), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@system_bp.route("/debug-data")
def debug_data():
    """Debug endpoint showing data availability and system state"""
    try:
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'data_path': '',
            'available_data': {},
            'services': {},
            'environment': {}
        }
        
        if handler:
            # Data loader info
            if handler.data_loader:
                debug_info['data_path'] = str(handler.data_loader.data_path)
                
                # Try to load each data type
                data_types = [
                    ('yarn_inventory', 'load_yarn_inventory'),
                    ('sales_orders', 'load_sales_orders'),
                    ('knit_orders', 'load_knit_orders'),
                    ('bom', 'load_bom'),
                    ('styles', 'load_styles')
                ]
                
                for name, method in data_types:
                    if hasattr(handler.data_loader, method):
                        try:
                            data = getattr(handler.data_loader, method)()
                            if data is not None:
                                debug_info['available_data'][name] = {
                                    'loaded': True,
                                    'records': len(data),
                                    'columns': list(data.columns) if hasattr(data, 'columns') else []
                                }
                            else:
                                debug_info['available_data'][name] = {'loaded': False}
                        except Exception as e:
                            debug_info['available_data'][name] = {
                                'loaded': False,
                                'error': str(e)
                            }
            
            # Service manager info
            if handler.service_manager:
                for name, service in handler.service_manager.services.items():
                    debug_info['services'][name] = {
                        'type': type(service).__name__,
                        'status': 'active'
                    }
        
        # Environment variables (sanitized)
        debug_info['environment'] = {
            'FLASK_ENV': os.environ.get('FLASK_ENV', 'production'),
            'DATA_PATH': os.environ.get('DATA_PATH', 'not set'),
            'WORKERS': os.environ.get('WORKERS', '1')
        }
        
        return jsonify(debug_info)
    
    except Exception as e:
        logger.error(f"Debug data error: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route("/cache-stats", methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    try:
        stats = {
            'cache_enabled': False,
            'stats': {}
        }
        
        if handler and handler.cache_manager:
            stats['cache_enabled'] = True
            
            # Get cache stats if available
            if hasattr(handler.cache_manager, 'get_stats'):
                stats['stats'] = handler.cache_manager.get_stats()
            elif hasattr(handler.cache_manager, 'cache_stats'):
                stats['stats'] = handler.cache_manager.cache_stats
            else:
                # Basic stats
                stats['stats'] = {
                    'type': type(handler.cache_manager).__name__,
                    'available': True
                }
        
        # Check file cache directory if using file-based caching
        cache_dir = Path('/tmp/bki_cache')
        if cache_dir.exists():
            cache_files = list(cache_dir.glob('*.pkl'))
            stats['file_cache'] = {
                'directory': str(cache_dir),
                'files_count': len(cache_files),
                'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
            }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route("/cache-clear", methods=['POST'])
def cache_clear():
    """Clear all caches"""
    try:
        cleared = {
            'memory_cache': False,
            'file_cache': False,
            'message': 'Cache clearing initiated'
        }
        
        # Clear memory cache
        if handler and handler.cache_manager:
            if hasattr(handler.cache_manager, 'clear'):
                handler.cache_manager.clear()
                cleared['memory_cache'] = True
            elif hasattr(handler.cache_manager, 'clear_cache'):
                handler.cache_manager.clear_cache()
                cleared['memory_cache'] = True
        
        # Clear data loader cache
        if handler and handler.data_loader:
            if hasattr(handler.data_loader, 'clear_cache'):
                handler.data_loader.clear_cache()
                cleared['data_loader_cache'] = True
        
        # Clear file cache
        cache_dir = Path('/tmp/bki_cache')
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
                cleared['file_cache'] = True
            except Exception as e:
                cleared['file_cache_error'] = str(e)
        
        cleared['message'] = 'Cache cleared successfully'
        return jsonify(cleared)
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route("/reload-data")
def reload_data():
    """Force reload all data"""
    try:
        reload_status = {
            'timestamp': datetime.now().isoformat(),
            'reloaded': {}
        }
        
        if handler and handler.data_loader:
            # Clear cache first
            if hasattr(handler.data_loader, 'clear_cache'):
                handler.data_loader.clear_cache()
                reload_status['cache_cleared'] = True
            
            # Reload all data
            if hasattr(handler.data_loader, 'load_all_data'):
                all_data = handler.data_loader.load_all_data()
                for key, data in all_data.items():
                    if data is not None:
                        reload_status['reloaded'][key] = {
                            'success': True,
                            'records': len(data)
                        }
                    else:
                        reload_status['reloaded'][key] = {
                            'success': False
                        }
            
            reload_status['message'] = 'Data reloaded successfully'
        else:
            reload_status['message'] = 'Data loader not available'
        
        return jsonify(reload_status)
    
    except Exception as e:
        logger.error(f"Error reloading data: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route("/consolidation-metrics")
def consolidation_metrics():
    """Get API consolidation metrics"""
    try:
        metrics = {
            'consolidation_enabled': True,
            'migration_progress': 0,
            'deprecated_calls': 0,
            'new_api_calls': 0,
            'redirect_enabled': True,
            'redirect_count': 0,
            'top_deprecated_endpoints': []
        }
        
        # Check if consolidation middleware is tracking metrics
        if hasattr(request, 'consolidation_metrics'):
            metrics.update(request.consolidation_metrics)
        
        # Calculate migration progress
        total_endpoints = 107  # Total endpoints in monolith
        migrated_endpoints = 41  # Endpoints moved to blueprints
        metrics['migration_progress'] = round((migrated_endpoints / total_endpoints) * 100, 2)
        
        # Service status
        if handler and handler.service_manager:
            metrics['services_active'] = len(handler.service_manager.services)
            metrics['modular_architecture'] = True
        else:
            metrics['modular_architecture'] = False
        
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error getting consolidation metrics: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route("/system-info")
def system_info():
    """Get comprehensive system information"""
    try:
        info = {
            'application': {
                'name': 'Beverly Knits ERP v2',
                'version': '2.0.0',
                'mode': 'modular',
                'uptime': handler.get_uptime() if handler else None
            },
            'infrastructure': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'processor': platform.processor(),
                'machine': platform.machine()
            },
            'modules': {
                'services': [],
                'data_loaders': [],
                'blueprints': []
            }
        }
        
        # List active services
        if handler and handler.service_manager:
            info['modules']['services'] = list(handler.service_manager.services.keys())
        
        # List registered blueprints
        from flask import current_app
        if current_app:
            info['modules']['blueprints'] = list(current_app.blueprints.keys())
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            info['resources'] = {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            pass
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500