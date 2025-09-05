"""
Beverly Knits ERP - Service Registry
Central registry for all modularized services
"""

import logging
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseService(ABC):
    """Base class for all services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize service-specific resources"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            'service': self.__class__.__name__,
            'status': 'running',
            'config': self.config
        }

class ServiceRegistry:
    """Central registry for all services"""
    
    _instance = None
    _services: Dict[str, BaseService] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
        return cls._instance
    
    def register(self, name: str, service: BaseService):
        """Register a service"""
        if name in self._services:
            logger.warning(f"Service {name} already registered, overwriting")
        
        self._services[name] = service
        logger.info(f"Service {name} registered successfully")
    
    def get(self, name: str) -> Optional[BaseService]:
        """Get a registered service"""
        return self._services.get(name)
    
    def get_all(self) -> Dict[str, BaseService]:
        """Get all registered services"""
        return self._services
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {}
        for name, service in self._services.items():
            try:
                health[name] = service.health_check()
            except Exception as e:
                health[name] = {'status': 'error', 'error': str(e)}
        
        return health
    
    def shutdown(self):
        """Shutdown all services"""
        for name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    service.shutdown()
                logger.info(f"Service {name} shut down")
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
    
    def initialize_all(self):
        """Initialize all services with dependencies"""
        # This can be extended to handle dependency injection
        for name, service in self._services.items():
            logger.info(f"Initializing service: {name}")

# Global registry instance
registry = ServiceRegistry()

def register_service(name: str):
    """Decorator to register a service"""
    def decorator(cls: Type[BaseService]):
        instance = cls()
        registry.register(name, instance)
        return cls
    return decorator

def get_service(name: str) -> Optional[BaseService]:
    """Get a service from the registry"""
    return registry.get(name)