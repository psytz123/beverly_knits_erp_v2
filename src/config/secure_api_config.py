"""
Secure API Configuration Management for eFab.ai Integration
Implements secure credential storage and retrieval patterns
"""

import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import hashlib

logger = logging.getLogger(__name__)


class EFabAPIConfig:
    """
    Secure configuration for eFab API access
    Manages credentials from multiple sources with encryption
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'base_url': 'https://efab.bkiapps.com',
        'session_timeout': 3600,  # 1 hour
        'retry_count': 3,
        'retry_delay': 1.0,
        'retry_backoff': 2.0,
        'circuit_breaker_threshold': 5,
        'circuit_breaker_timeout': 60,
        'max_parallel_requests': 5,
        'cache_ttl_yarn': 900,  # 15 minutes
        'cache_ttl_orders': 300,  # 5 minutes
        'cache_ttl_styles': 3600,  # 1 hour
        'connection_timeout': 30,
        'read_timeout': 60,
        'enable_monitoring': True,
        'enable_fallback': True,
        'enable_cache': True
    }
    
    # API endpoint paths
    API_ENDPOINTS = {
        'yarn_active': '/api/yarn/active',
        'greige_g00': '/api/greige/g00',
        'greige_g02': '/api/greige/g02',
        'finished_i01': '/api/finished/i01',
        'finished_f01': '/api/finished/f01',
        'yarn_po': '/api/yarn-po',
        'knit_orders': '/fabric/knitorder/list',
        'styles': '/api/styles',
        'yarn_expected': '/api/report/yarn_expected',
        'sales_activity': '/api/report/sales_activity',
        'yarn_demand': '/api/report/yarn_demand',
        'yarn_demand_ko': '/api/report/yarn_demand_ko',
        'sales_order_plan': '/api/sales-order/plan/list',
        'login': '/api/auth/login',
        'logout': '/api/auth/logout',
        'health': '/api/health'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._config_cache = None
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Check for environment variable first
        if os.getenv('EFAB_CONFIG_PATH'):
            return os.getenv('EFAB_CONFIG_PATH')
        
        # Default to project config directory
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / 'config'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'efab_api_config.json')
    
    def _get_or_create_encryption_key(self) -> bytes:
        """
        Get or create encryption key for sensitive data
        
        Returns:
            Encryption key bytes
        """
        # Try to get from environment variable
        key_str = os.getenv('EFAB_ENCRYPTION_KEY')
        if key_str:
            try:
                return base64.urlsafe_b64decode(key_str.encode())
            except Exception:
                logger.warning("Invalid encryption key in environment, generating new one")
        
        # Try to load from file
        key_file = Path.home() / '.efab' / 'encryption.key'
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load encryption key from file: {e}")
        
        # Generate new key
        key = Fernet.generate_key()
        
        # Try to save for future use
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info("Generated and saved new encryption key")
        except Exception as e:
            logger.warning(f"Could not save encryption key: {e}")
        
        return key
    
    def get_credentials(self) -> Dict[str, any]:
        """
        Get credentials from secure sources
        Priority order:
        1. Environment variables
        2. Encrypted configuration file
        3. Default values
        
        Returns:
            Dictionary containing configuration and credentials
        """
        config = self.DEFAULT_CONFIG.copy()
        
        # Load from environment variables (highest priority)
        env_mapping = {
            'EFAB_BASE_URL': 'base_url',
            'EFAB_USERNAME': 'username',
            'EFAB_PASSWORD': 'password',
            'EFAB_SESSION_TIMEOUT': ('session_timeout', int),
            'EFAB_RETRY_COUNT': ('retry_count', int),
            'EFAB_RETRY_DELAY': ('retry_delay', float),
            'EFAB_RETRY_BACKOFF': ('retry_backoff', float),
            'EFAB_CIRCUIT_BREAKER_THRESHOLD': ('circuit_breaker_threshold', int),
            'EFAB_CIRCUIT_BREAKER_TIMEOUT': ('circuit_breaker_timeout', int),
            'EFAB_MAX_PARALLEL_REQUESTS': ('max_parallel_requests', int),
            'EFAB_CACHE_TTL_YARN': ('cache_ttl_yarn', int),
            'EFAB_CACHE_TTL_ORDERS': ('cache_ttl_orders', int),
            'EFAB_CONNECTION_TIMEOUT': ('connection_timeout', int),
            'EFAB_READ_TIMEOUT': ('read_timeout', int),
            'EFAB_ENABLE_MONITORING': ('enable_monitoring', lambda x: x.lower() == 'true'),
            'EFAB_ENABLE_FALLBACK': ('enable_fallback', lambda x: x.lower() == 'true'),
            'EFAB_ENABLE_CACHE': ('enable_cache', lambda x: x.lower() == 'true')
        }
        
        for env_var, mapping in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                if isinstance(mapping, tuple):
                    key, converter = mapping
                    try:
                        config[key] = converter(value)
                    except ValueError:
                        logger.warning(f"Invalid value for {env_var}: {value}")
                else:
                    config[mapping] = value
        
        # Load from configuration file
        file_config = self._load_config_file()
        if file_config:
            # File config has lower priority than env vars
            for key, value in file_config.items():
                if key not in config or config[key] == self.DEFAULT_CONFIG.get(key):
                    config[key] = value
        
        # Validate required credentials
        if not config.get('username') or not config.get('password'):
            logger.warning("API credentials not found. API integration will be disabled.")
            config['api_enabled'] = False
        else:
            config['api_enabled'] = True
        
        # Add endpoint URLs
        config['endpoints'] = {
            name: config['base_url'] + path
            for name, path in self.API_ENDPOINTS.items()
        }
        
        return config
    
    def _load_config_file(self) -> Optional[Dict]:
        """
        Load configuration from encrypted file
        
        Returns:
            Configuration dictionary or None
        """
        if self._config_cache:
            return self._config_cache
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Decrypt sensitive fields if they're encrypted
            if data.get('encrypted'):
                for field in ['password', 'api_key', 'secret']:
                    if field in data and isinstance(data[field], str) and data[field].startswith('ENC:'):
                        encrypted_value = data[field][4:]  # Remove 'ENC:' prefix
                        try:
                            decrypted = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
                            data[field] = decrypted
                        except Exception as e:
                            logger.error(f"Failed to decrypt {field}: {e}")
            
            self._config_cache = data
            return data
            
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return None
    
    def save_credentials(self, credentials: Dict[str, any], encrypt_sensitive: bool = True):
        """
        Save credentials to configuration file
        
        Args:
            credentials: Credentials dictionary
            encrypt_sensitive: Whether to encrypt sensitive fields
        """
        try:
            config = credentials.copy()
            
            # Encrypt sensitive fields
            if encrypt_sensitive:
                config['encrypted'] = True
                for field in ['password', 'api_key', 'secret']:
                    if field in config and config[field]:
                        encrypted = self.cipher_suite.encrypt(config[field].encode()).decode()
                        config[field] = f"ENC:{encrypted}"
            
            # Ensure directory exists
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Clear cache
            self._config_cache = None
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def validate_config(self, config: Dict) -> bool:
        """
        Validate configuration completeness
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        required_fields = ['base_url', 'username', 'password']
        
        for field in required_fields:
            if not config.get(field):
                logger.error(f"Missing required configuration field: {field}")
                return False
        
        # Validate URL format
        if not config['base_url'].startswith(('http://', 'https://')):
            logger.error("Invalid base_url format. Must start with http:// or https://")
            return False
        
        # Validate numeric fields
        numeric_fields = [
            'session_timeout', 'retry_count', 'circuit_breaker_threshold',
            'max_parallel_requests', 'connection_timeout', 'read_timeout'
        ]
        
        for field in numeric_fields:
            if field in config:
                try:
                    value = int(config[field])
                    if value <= 0:
                        logger.error(f"Invalid {field}: must be positive integer")
                        return False
                except (TypeError, ValueError):
                    logger.error(f"Invalid {field}: must be numeric")
                    return False
        
        return True
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """
        Get feature flags for API integration
        
        Returns:
            Dictionary of feature flags
        """
        config = self.get_credentials()
        
        return {
            'api_enabled': config.get('api_enabled', False),
            'enable_monitoring': config.get('enable_monitoring', True),
            'enable_fallback': config.get('enable_fallback', True),
            'enable_cache': config.get('enable_cache', True),
            'enable_parallel_loading': config.get('max_parallel_requests', 1) > 1,
            'enable_circuit_breaker': config.get('circuit_breaker_threshold', 0) > 0
        }
    
    def get_retry_config(self) -> Dict:
        """
        Get retry configuration
        
        Returns:
            Retry configuration dictionary
        """
        config = self.get_credentials()
        
        return {
            'stop_max_attempt_number': config.get('retry_count', 3),
            'wait_fixed': int(config.get('retry_delay', 1) * 1000),  # Convert to ms
            'wait_exponential_multiplier': int(config.get('retry_backoff', 2) * 1000),
            'wait_exponential_max': 30000  # Max 30 seconds
        }
    
    def get_cache_config(self) -> Dict[str, int]:
        """
        Get cache TTL configuration for different data types
        
        Returns:
            Dictionary mapping data type to TTL in seconds
        """
        config = self.get_credentials()
        
        return {
            'yarn': config.get('cache_ttl_yarn', 900),
            'orders': config.get('cache_ttl_orders', 300),
            'styles': config.get('cache_ttl_styles', 3600),
            'greige': config.get('cache_ttl_orders', 300),
            'finished': config.get('cache_ttl_orders', 300),
            'po': config.get('cache_ttl_orders', 300),
            'sales': config.get('cache_ttl_yarn', 900)
        }
    
    def get_timeout_config(self) -> tuple:
        """
        Get timeout configuration
        
        Returns:
            Tuple of (connection_timeout, read_timeout) in seconds
        """
        config = self.get_credentials()
        
        return (
            config.get('connection_timeout', 30),
            config.get('read_timeout', 60)
        )


def get_api_config() -> EFabAPIConfig:
    """
    Factory function to get API configuration instance
    
    Returns:
        Configured EFabAPIConfig instance
    """
    return EFabAPIConfig()


if __name__ == "__main__":
    # Test configuration loading
    config_manager = get_api_config()
    config = config_manager.get_credentials()
    
    print("API Configuration Status:")
    print(f"  Base URL: {config.get('base_url')}")
    print(f"  API Enabled: {config.get('api_enabled')}")
    print(f"  Username Set: {'username' in config and bool(config['username'])}")
    print(f"  Password Set: {'password' in config and bool(config['password'])}")
    print(f"  Session Timeout: {config.get('session_timeout')} seconds")
    print(f"  Retry Count: {config.get('retry_count')}")
    print(f"  Cache Enabled: {config.get('enable_cache')}")
    
    print("\nFeature Flags:")
    for flag, value in config_manager.get_feature_flags().items():
        print(f"  {flag}: {value}")
    
    print("\nCache Configuration:")
    for data_type, ttl in config_manager.get_cache_config().items():
        print(f"  {data_type}: {ttl} seconds")