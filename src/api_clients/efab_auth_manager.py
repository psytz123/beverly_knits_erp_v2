"""
eFab.ai Authentication Manager
Manages authentication lifecycle, session management, and token refresh
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuthStatus(Enum):
    """Authentication status enumeration"""
    NOT_AUTHENTICATED = "not_authenticated"
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    REFRESHING = "refreshing"
    FAILED = "failed"


@dataclass
class SessionInfo:
    """Session information container"""
    token: str
    created_at: datetime
    expires_at: datetime
    username: str
    user_id: Optional[str] = None
    refresh_token: Optional[str] = None
    permissions: Optional[list] = None


class EFabAuthManager:
    """
    Manages authentication lifecycle for eFab.ai API
    Handles login, session refresh, and token management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize authentication manager
        
        Args:
            config: API configuration dictionary
        """
        self.config = config
        self.base_url = config.get('base_url', 'https://efab.bkiapps.com')
        self.username = config.get('username')
        self.password = config.get('password')
        self.session_timeout = config.get('session_timeout', 3600)
        
        # Session management
        self.session_info: Optional[SessionInfo] = None
        self.auth_status = AuthStatus.NOT_AUTHENTICATED
        self._refresh_lock = asyncio.Lock()
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # Session refresh parameters
        self.refresh_before_expiry = 300  # Refresh 5 minutes before expiry
        self.max_refresh_attempts = 3
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize HTTP session and authenticate"""
        if not self._http_session:
            timeout = aiohttp.ClientTimeout(
                total=60,
                connect=self.config.get('connection_timeout', 30),
                sock_read=self.config.get('read_timeout', 60)
            )
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Attempt initial authentication
        if self.username and self.password:
            await self.authenticate()
    
    async def cleanup(self):
        """Clean up resources"""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
    
    async def authenticate(self) -> str:
        """
        Authenticate with eFab.ai API
        
        Returns:
            Session token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.username or not self.password:
            raise AuthenticationError("Username and password are required")
        
        async with self._refresh_lock:
            try:
                logger.info(f"Authenticating user: {self.username}")
                self.auth_status = AuthStatus.REFRESHING
                
                # Prepare authentication request
                auth_url = f"{self.base_url}/api/auth/login"
                auth_data = {
                    'username': self.username,
                    'password': self.password
                }
                
                # Send authentication request
                if not self._http_session:
                    await self.initialize()
                
                async with self._http_session.post(
                    auth_url,
                    json=auth_data,
                    ssl=True
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError:
                            # Fallback for non-JSON response
                            data = {'token': response_text.strip()}
                        
                        # Extract session information
                        token = data.get('token') or data.get('session_token')
                        if not token:
                            # Check if token is in cookies
                            token = response.cookies.get('dancer.session')
                        
                        if not token:
                            raise AuthenticationError("No session token received")
                        
                        # Create session info
                        now = datetime.now()
                        self.session_info = SessionInfo(
                            token=str(token),
                            created_at=now,
                            expires_at=now + timedelta(seconds=self.session_timeout),
                            username=self.username,
                            user_id=data.get('user_id'),
                            refresh_token=data.get('refresh_token'),
                            permissions=data.get('permissions', [])
                        )
                        
                        self.auth_status = AuthStatus.AUTHENTICATED
                        logger.info("Authentication successful")
                        
                        # Schedule automatic refresh
                        asyncio.create_task(self._schedule_refresh())
                        
                        return self.session_info.token
                    
                    elif response.status == 401:
                        self.auth_status = AuthStatus.FAILED
                        raise AuthenticationError(f"Invalid credentials: {response_text}")
                    
                    else:
                        self.auth_status = AuthStatus.FAILED
                        raise AuthenticationError(
                            f"Authentication failed with status {response.status}: {response_text}"
                        )
                        
            except aiohttp.ClientError as e:
                self.auth_status = AuthStatus.FAILED
                logger.error(f"Network error during authentication: {e}")
                raise AuthenticationError(f"Network error: {e}")
            
            except Exception as e:
                self.auth_status = AuthStatus.FAILED
                logger.error(f"Unexpected error during authentication: {e}")
                raise
    
    async def refresh_session(self) -> bool:
        """
        Refresh the current session before expiry
        
        Returns:
            True if refresh successful, False otherwise
        """
        if not self.session_info:
            logger.warning("No session to refresh")
            return False
        
        async with self._refresh_lock:
            try:
                logger.info("Refreshing session")
                self.auth_status = AuthStatus.REFRESHING
                
                # If we have a refresh token, use it
                if self.session_info.refresh_token:
                    success = await self._refresh_with_token()
                    if success:
                        return True
                
                # Otherwise, re-authenticate
                await self.authenticate()
                return True
                
            except Exception as e:
                logger.error(f"Failed to refresh session: {e}")
                self.auth_status = AuthStatus.EXPIRED
                return False
    
    async def _refresh_with_token(self) -> bool:
        """
        Refresh session using refresh token
        
        Returns:
            True if successful
        """
        if not self.session_info or not self.session_info.refresh_token:
            return False
        
        try:
            refresh_url = f"{self.base_url}/api/auth/refresh"
            refresh_data = {
                'refresh_token': self.session_info.refresh_token
            }
            
            async with self._http_session.post(
                refresh_url,
                json=refresh_data,
                headers=self.get_auth_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Update session info
                    now = datetime.now()
                    self.session_info.token = data.get('token', self.session_info.token)
                    self.session_info.created_at = now
                    self.session_info.expires_at = now + timedelta(seconds=self.session_timeout)
                    self.session_info.refresh_token = data.get('refresh_token', self.session_info.refresh_token)
                    
                    self.auth_status = AuthStatus.AUTHENTICATED
                    logger.info("Session refreshed successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to refresh with token: {e}")
        
        return False
    
    async def _schedule_refresh(self):
        """Schedule automatic session refresh"""
        if not self.session_info:
            return
        
        while self.auth_status == AuthStatus.AUTHENTICATED:
            try:
                # Calculate time until refresh needed
                time_until_expiry = (self.session_info.expires_at - datetime.now()).total_seconds()
                time_until_refresh = max(0, time_until_expiry - self.refresh_before_expiry)
                
                if time_until_refresh > 0:
                    logger.debug(f"Scheduling refresh in {time_until_refresh} seconds")
                    await asyncio.sleep(time_until_refresh)
                
                # Check if still authenticated
                if self.auth_status == AuthStatus.AUTHENTICATED:
                    await self.refresh_session()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh scheduler: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def is_session_valid(self) -> bool:
        """
        Check if current session is valid
        
        Returns:
            True if session is valid and not expired
        """
        if not self.session_info:
            return False
        
        if self.auth_status != AuthStatus.AUTHENTICATED:
            return False
        
        # Check expiry
        if datetime.now() >= self.session_info.expires_at:
            self.auth_status = AuthStatus.EXPIRED
            return False
        
        return True
    
    async def ensure_authenticated(self) -> str:
        """
        Ensure we have a valid authentication token
        
        Returns:
            Valid session token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Check if current session is valid
        if self.is_session_valid():
            return self.session_info.token
        
        # Check if refresh is needed
        if self.session_info and datetime.now() < self.session_info.expires_at:
            # Session exists but might need refresh soon
            time_until_expiry = (self.session_info.expires_at - datetime.now()).total_seconds()
            if time_until_expiry < self.refresh_before_expiry:
                if await self.refresh_session():
                    return self.session_info.token
        
        # Need to authenticate
        return await self.authenticate()
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests
        
        Returns:
            Dictionary of headers including authentication
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.session_info and self.session_info.token:
            # Try both cookie and Authorization header formats
            headers['Cookie'] = f"dancer.session={self.session_info.token}"
            headers['Authorization'] = f"Bearer {self.session_info.token}"
        
        return headers
    
    async def logout(self):
        """Logout and clear session"""
        try:
            if self.session_info and self._http_session:
                logout_url = f"{self.base_url}/api/auth/logout"
                
                async with self._http_session.post(
                    logout_url,
                    headers=self.get_auth_headers()
                ) as response:
                    if response.status == 200:
                        logger.info("Logged out successfully")
                    else:
                        logger.warning(f"Logout returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Error during logout: {e}")
        
        finally:
            # Clear session info regardless
            self.session_info = None
            self.auth_status = AuthStatus.NOT_AUTHENTICATED
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current session information
        
        Returns:
            Session info dictionary or None
        """
        if not self.session_info:
            return None
        
        return {
            'username': self.session_info.username,
            'user_id': self.session_info.user_id,
            'created_at': self.session_info.created_at.isoformat(),
            'expires_at': self.session_info.expires_at.isoformat(),
            'time_remaining': max(0, (self.session_info.expires_at - datetime.now()).total_seconds()),
            'status': self.auth_status.value,
            'permissions': self.session_info.permissions or []
        }
    
    def __repr__(self) -> str:
        """String representation"""
        status = self.auth_status.value
        username = self.username or "Not configured"
        valid = "Valid" if self.is_session_valid() else "Invalid"
        return f"<EFabAuthManager: {username}, Status: {status}, Session: {valid}>"


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


async def test_auth_manager():
    """Test authentication manager functionality"""
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from config.secure_api_config import get_api_config
    
    # Get configuration
    config_manager = get_api_config()
    config = config_manager.get_credentials()
    
    if not config.get('username') or not config.get('password'):
        print("No credentials configured. Set EFAB_USERNAME and EFAB_PASSWORD environment variables.")
        return
    
    # Test authentication
    auth_manager = EFabAuthManager(config)
    
    try:
        async with auth_manager:
            # Authenticate
            token = await auth_manager.authenticate()
            print(f"Authentication successful. Token: {token[:20]}...")
            
            # Check session
            print(f"Session valid: {auth_manager.is_session_valid()}")
            
            # Get session info
            info = auth_manager.get_session_info()
            print(f"Session info: {json.dumps(info, indent=2)}")
            
            # Test refresh
            print("Testing session refresh...")
            success = await auth_manager.refresh_session()
            print(f"Refresh successful: {success}")
            
            # Logout
            await auth_manager.logout()
            print("Logged out successfully")
            
    except AuthenticationError as e:
        print(f"Authentication error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_auth_manager())