"""
eFab API Connection Manager with persistent connections and auto-reconnect
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import aiohttp
from enum import Enum
from dataclasses import dataclass, field
import signal
import sys

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class ConnectionStats:
    """Connection statistics"""
    state: ConnectionState = ConnectionState.DISCONNECTED
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    reconnect_attempts: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    uptime_seconds: float = 0
    last_error: Optional[str] = None

class PersistentConnectionPool:
    """
    Manages persistent connections with keep-alive and auto-reconnect
    """
    
    def __init__(
        self,
        base_url: str,
        max_connections: int = 10,
        keepalive_interval: int = 30,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = -1,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ):
        """
        Initialize connection pool
        
        Args:
            base_url: Base URL for API
            max_connections: Maximum concurrent connections
            keepalive_interval: Seconds between keep-alive pings
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite)
            connection_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
        """
        self.base_url = base_url
        self.max_connections = max_connections
        self.keepalive_interval = keepalive_interval
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Connection settings
        self.timeout = aiohttp.ClientTimeout(
            total=None,  # No total timeout for persistent connections
            connect=connection_timeout,
            sock_read=read_timeout
        )
        
        # Connector with connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=30
        )
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = ConnectionStats()
        
        # Tasks
        self._keepalive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        """Start connection pool and background tasks"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self.stats.state = ConnectionState.CONNECTING
            
            # Create session
            await self._create_session()
            
            # Start background tasks
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            logger.info(f"Connection pool started for {self.base_url}")
    
    async def stop(self):
        """Stop connection pool and cleanup"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            # Cancel background tasks
            for task in [self._keepalive_task, self._reconnect_task, self._monitor_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close session
            await self._close_session()
            
            logger.info("Connection pool stopped")
    
    async def _create_session(self):
        """Create aiohttp session"""
        try:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                connector_owner=False,  # Don't close connector with session
                headers={
                    'Connection': 'keep-alive',
                    'Keep-Alive': f'timeout={self.keepalive_interval}'
                }
            )
            
            self.stats.state = ConnectionState.CONNECTED
            self.stats.connected_at = datetime.now()
            self.stats.reconnect_attempts = 0
            
            if self.on_connect:
                await self.on_connect()
            
            logger.info("Session created successfully")
            
        except Exception as e:
            self.stats.state = ConnectionState.ERROR
            self.stats.last_error = str(e)
            logger.error(f"Failed to create session: {e}")
            await self._schedule_reconnect()
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                self.session = None
                self.stats.state = ConnectionState.DISCONNECTED
                self.stats.disconnected_at = datetime.now()
                
                if self.on_disconnect:
                    await self.on_disconnect()
    
    async def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if not self._running:
            return
        
        if self.max_reconnect_attempts > 0 and self.stats.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.stats.state = ConnectionState.ERROR
            return
        
        self.stats.state = ConnectionState.RECONNECTING
        self.stats.reconnect_attempts += 1
        
        logger.info(f"Scheduling reconnection attempt {self.stats.reconnect_attempts} in {self.reconnect_interval} seconds")
        
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        
        self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self):
        """Perform reconnection"""
        await asyncio.sleep(self.reconnect_interval)
        
        if not self._running:
            return
        
        logger.info(f"Reconnection attempt {self.stats.reconnect_attempts}")
        
        # Close existing session
        await self._close_session()
        
        # Create new session
        await self._create_session()
    
    async def _keepalive_loop(self):
        """Keep-alive loop to maintain connection"""
        while self._running:
            try:
                await asyncio.sleep(self.keepalive_interval)
                
                if self.session and self.stats.state == ConnectionState.CONNECTED:
                    # Send keep-alive ping
                    await self.ping()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
    
    async def _monitor_loop(self):
        """Monitor connection health"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Update uptime
                if self.stats.state == ConnectionState.CONNECTED and self.stats.connected_at:
                    self.stats.uptime_seconds = (datetime.now() - self.stats.connected_at).total_seconds()
                
                # Check for stale connection
                if self.stats.last_activity:
                    idle_time = (datetime.now() - self.stats.last_activity).total_seconds()
                    if idle_time > 300:  # 5 minutes idle
                        logger.warning(f"Connection idle for {idle_time:.0f} seconds")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    async def ping(self) -> bool:
        """
        Send ping to keep connection alive
        
        Returns:
            True if ping successful
        """
        if not self.session:
            return False
        
        try:
            async with self.session.head(self.base_url) as response:
                self.stats.last_activity = datetime.now()
                return response.status < 500
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            self.stats.last_error = str(e)
            await self._schedule_reconnect()
            return False
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request through connection pool
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            Response object
            
        Raises:
            Exception if not connected
        """
        if not self.session:
            raise RuntimeError("Not connected")
        
        if self.stats.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Connection state: {self.stats.state.value}")
        
        try:
            self.stats.total_requests += 1
            self.stats.last_activity = datetime.now()
            
            response = await self.session.request(method, url, **kwargs)
            return response
            
        except Exception as e:
            self.stats.failed_requests += 1
            self.stats.last_error = str(e)
            
            if self.on_error:
                await self.on_error(e)
            
            # Check if we need to reconnect
            if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                await self._schedule_reconnect()
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'state': self.stats.state.value,
            'connected_at': self.stats.connected_at.isoformat() if self.stats.connected_at else None,
            'uptime_seconds': self.stats.uptime_seconds,
            'last_activity': self.stats.last_activity.isoformat() if self.stats.last_activity else None,
            'reconnect_attempts': self.stats.reconnect_attempts,
            'total_requests': self.stats.total_requests,
            'failed_requests': self.stats.failed_requests,
            'error_rate': self.stats.failed_requests / max(1, self.stats.total_requests),
            'last_error': self.stats.last_error,
            'active_connections': len(self.connector._acquired) if self.connector else 0,
            'available_connections': self.connector._limit - len(self.connector._acquired) if self.connector else 0
        }
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.stats.state == ConnectionState.CONNECTED and self.session is not None


class EFabPersistentClient:
    """
    High-level client with persistent connection management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize persistent client
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_url = config.get('base_url', 'https://efab.bkiapps.com')
        
        # Create connection pool
        self.pool = PersistentConnectionPool(
            base_url=self.base_url,
            max_connections=config.get('max_connections', 10),
            keepalive_interval=config.get('keepalive_interval', 30),
            reconnect_interval=config.get('reconnect_interval', 5),
            max_reconnect_attempts=config.get('max_reconnect_attempts', -1),
            connection_timeout=config.get('connection_timeout', 30),
            read_timeout=config.get('read_timeout', 60)
        )
        
        # Set callbacks
        self.pool.on_connect = self._on_connect
        self.pool.on_disconnect = self._on_disconnect
        self.pool.on_error = self._on_error
        
        # Authentication headers (to be set after auth)
        self.auth_headers: Dict[str, str] = {}
        
        # Shutdown handler
        self._setup_shutdown_handler()
    
    def _setup_shutdown_handler(self):
        """Setup graceful shutdown handler"""
        def shutdown_handler(signum, frame):
            logger.info("Received shutdown signal, closing connections...")
            asyncio.create_task(self.stop())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    
    async def _on_connect(self):
        """Handle connection established"""
        logger.info("eFab API connected successfully")
        # Could trigger authentication here if needed
    
    async def _on_disconnect(self):
        """Handle connection lost"""
        logger.warning("eFab API connection lost")
    
    async def _on_error(self, error: Exception):
        """Handle connection error"""
        logger.error(f"eFab API connection error: {error}")
    
    async def start(self):
        """Start persistent client"""
        await self.pool.start()
        logger.info("eFab persistent client started")
    
    async def stop(self):
        """Stop persistent client"""
        await self.pool.stop()
        logger.info("eFab persistent client stopped")
    
    async def set_auth_headers(self, headers: Dict[str, str]):
        """Set authentication headers"""
        self.auth_headers = headers
    
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make authenticated request
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response data
        """
        # Build full URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = f"{self.base_url}{endpoint}"
        
        # Add auth headers
        headers = kwargs.get('headers', {})
        headers.update(self.auth_headers)
        kwargs['headers'] = headers
        
        # Make request through pool
        async with await self.pool.request(method, url, **kwargs) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=text
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            'connection_pool': self.pool.get_stats(),
            'base_url': self.base_url,
            'authenticated': bool(self.auth_headers)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status
        """
        health = {
            'healthy': False,
            'connection': self.pool.get_stats(),
            'checks': {}
        }
        
        # Check connection
        health['checks']['connected'] = self.pool.is_connected()
        
        # Try ping
        if self.pool.is_connected():
            health['checks']['ping'] = await self.pool.ping()
        
        # Overall health
        health['healthy'] = all(health['checks'].values())
        
        return health


async def test_persistent_connection():
    """Test persistent connection manager"""
    import os
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.config.secure_api_config import get_api_config
    from src.api_clients.efab_auth_manager import EFabAuthManager
    
    print("Testing eFab Persistent Connection")
    print("=" * 50)
    
    # Get configuration
    config_manager = get_api_config()
    config = config_manager.get_credentials()
    
    if not config.get('api_enabled'):
        print("API not configured. Set EFAB_USERNAME and EFAB_PASSWORD environment variables.")
        return
    
    # Create persistent client
    client = EFabPersistentClient(config)
    
    # Start client
    await client.start()
    
    # Authenticate
    auth_manager = EFabAuthManager(config)
    await auth_manager.initialize()
    await auth_manager.authenticate()
    
    # Set auth headers
    await client.set_auth_headers(auth_manager.get_auth_headers())
    
    print("\nInitial Status:")
    status = client.get_status()
    print(f"  Connection State: {status['connection_pool']['state']}")
    print(f"  Authenticated: {status['authenticated']}")
    
    # Test requests
    print("\nTesting Requests:")
    for i in range(5):
        try:
            print(f"\n  Request {i+1}:")
            data = await client.request('GET', '/api/health')
            print(f"    Response: {data}")
            
            # Get status
            stats = client.pool.get_stats()
            print(f"    Total Requests: {stats['total_requests']}")
            print(f"    Failed Requests: {stats['failed_requests']}")
            print(f"    Active Connections: {stats['active_connections']}")
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Health check
    print("\nHealth Check:")
    health = await client.health_check()
    print(f"  Healthy: {health['healthy']}")
    print(f"  Checks: {health['checks']}")
    
    # Monitor for 30 seconds
    print("\nMonitoring for 30 seconds...")
    for i in range(6):
        await asyncio.sleep(5)
        stats = client.pool.get_stats()
        print(f"  [{i*5}s] State: {stats['state']}, Uptime: {stats['uptime_seconds']:.0f}s, Requests: {stats['total_requests']}")
    
    # Final status
    print("\nFinal Status:")
    final_status = client.get_status()
    pool_stats = final_status['connection_pool']
    print(f"  State: {pool_stats['state']}")
    print(f"  Uptime: {pool_stats['uptime_seconds']:.0f} seconds")
    print(f"  Total Requests: {pool_stats['total_requests']}")
    print(f"  Error Rate: {pool_stats['error_rate']:.2%}")
    
    # Cleanup
    await auth_manager.cleanup()
    await client.stop()
    
    print("\nTest completed")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_persistent_connection())