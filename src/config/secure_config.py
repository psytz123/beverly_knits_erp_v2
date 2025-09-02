#!/usr/bin/env python3
"""
Secure Configuration Module for Beverly Knits ERP
Loads database credentials from environment variables
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration management using environment variables"""
    
    @staticmethod
    def load_env_file(env_path: str = '.env') -> None:
        """Load environment variables from .env file if it exists"""
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    @staticmethod
    def get_database_config() -> Dict[str, Any]:
        """Get database configuration from environment variables"""
        # Try to load .env file if not in production
        if os.getenv('ENVIRONMENT', 'development') != 'production':
            SecureConfig.load_env_file()
        
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'beverly_knits_erp'),
            'user': os.getenv('DB_USER', 'erp_user'),
            'password': os.getenv('DB_PASSWORD', 'erp_password'),
            'connection_pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
            'max_overflow': int(os.getenv('DB_POOL_MAX_OVERFLOW', 20))
        }
    
    @staticmethod
    def get_api_config() -> Dict[str, Any]:
        """Get API configuration from environment variables"""
        return {
            'api_port': int(os.getenv('API_PORT', 5007)),
            'erp_port': int(os.getenv('ERP_PORT', 5006)),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
    
    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """Get data path configuration from environment variables"""
        return {
            'data_base_path': os.getenv('DATA_BASE_PATH', '/mnt/c/Users/psytz/sc data/ERP Data'),
            'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            'cache_ttl': int(os.getenv('CACHE_TTL', 300))
        }
    
    @staticmethod
    def get_unified_config() -> Dict[str, Any]:
        """Get complete unified configuration"""
        db_config = SecureConfig.get_database_config()
        api_config = SecureConfig.get_api_config()
        data_config = SecureConfig.get_data_config()
        
        return {
            'data_source': {
                'primary': 'database',
                'fallback': 'files',
                'enable_dual_source': True,
                'database': db_config,
                'files': {
                    'primary_path': data_config['data_base_path'],
                    'batch_size': 1000
                }
            },
            'api': api_config,
            'cache': {
                'enabled': data_config['cache_enabled'],
                'ttl': data_config['cache_ttl']
            }
        }
    
    @staticmethod
    def validate_config() -> bool:
        """Validate that required configuration is present"""
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            logger.warning(f"Missing environment variables: {missing}")
            logger.info("Using default values from fallback configuration")
            return False
        
        return True


# Convenience functions for backwards compatibility
def get_database_config():
    """Get database configuration"""
    return SecureConfig.get_database_config()

def get_api_config():
    """Get API configuration"""
    return SecureConfig.get_api_config()

def get_unified_config():
    """Get unified configuration"""
    return SecureConfig.get_unified_config()