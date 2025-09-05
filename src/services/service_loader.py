"""
Service Loader Wrapper
Ensures services load properly regardless of import path issues
Created: 2025-09-05
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def initialize_services(data_path=None):
    """
    Initialize services with proper path handling
    """
    try:
        # Try importing with proper paths set
        from services.service_integration import get_service_integration
        
        service_integration = get_service_integration(
            data_path=data_path,
            config={}
        )
        
        if service_integration and service_integration.is_initialized():
            print("[OK] Services initialized successfully via wrapper")
            return service_integration
        else:
            print("[WARNING] Service integration created but not fully initialized")
            return None
            
    except ImportError as e:
        print(f"[ERROR] Failed to import services: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        return None


def get_service_status():
    """
    Check status of service initialization
    """
    service_integration = initialize_services()
    
    if service_integration:
        try:
            health = service_integration.health_check()
            return {
                'status': 'OK',
                'services': health.get('services', {}),
                'overall': health.get('overall_status', 'UNKNOWN')
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    else:
        return {
            'status': 'NOT_INITIALIZED',
            'error': 'Service integration not available'
        }


# Export for easy use
__all__ = ['initialize_services', 'get_service_status']
