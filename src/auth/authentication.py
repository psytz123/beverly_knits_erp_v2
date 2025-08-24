#!/usr/bin/env python3
"""
Authentication Module for Beverly Knits ERP
Provides JWT-based authentication and role-based access control
"""

import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any
from flask import request, jsonify, current_app
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


class AuthManager:
    """Manages authentication and authorization for the ERP system"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = 'HS256'):
        """
        Initialize authentication manager
        
        Args:
            secret_key: Secret key for JWT encoding/decoding
            algorithm: Algorithm to use for JWT
        """
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY') or self._generate_secret_key()
        self.algorithm = algorithm
        self.token_expiry_hours = int(os.environ.get('JWT_EXPIRY_HOURS', 24))
        
        # Role definitions
        self.roles = {
            'admin': {
                'level': 100,
                'permissions': ['*']  # All permissions
            },
            'manager': {
                'level': 80,
                'permissions': [
                    'inventory.read', 'inventory.write',
                    'orders.read', 'orders.write',
                    'production.read', 'production.write',
                    'reports.read', 'reports.write',
                    'forecast.read'
                ]
            },
            'supervisor': {
                'level': 60,
                'permissions': [
                    'inventory.read', 'inventory.write',
                    'orders.read',
                    'production.read', 'production.write',
                    'reports.read'
                ]
            },
            'operator': {
                'level': 40,
                'permissions': [
                    'inventory.read',
                    'orders.read',
                    'production.read',
                    'reports.read'
                ]
            },
            'viewer': {
                'level': 20,
                'permissions': [
                    'inventory.read',
                    'reports.read'
                ]
            }
        }
        
        # API key management (for service-to-service auth)
        self.api_keys = {}
        self._load_api_keys()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key"""
        return secrets.token_urlsafe(32)
    
    def _load_api_keys(self):
        """Load API keys from environment or configuration"""
        # In production, these would be loaded from a secure store
        if os.environ.get('API_KEYS'):
            for key_entry in os.environ.get('API_KEYS', '').split(','):
                if ':' in key_entry:
                    name, key = key_entry.split(':', 1)
                    self.api_keys[key] = name
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """
        Hash a password using SHA-256
        
        Args:
            password: Plain text password
            salt: Optional salt (will be generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if not salt:
            salt = secrets.token_hex(16)
        
        # Combine password and salt
        salted_password = f"{password}{salt}".encode('utf-8')
        
        # Hash using SHA-256
        hashed = hashlib.sha256(salted_password).hexdigest()
        
        return hashed, salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password to verify
            hashed: Stored password hash
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        check_hash, _ = self.hash_password(password, salt)
        return check_hash == hashed
    
    def generate_token(self, user_id: str, username: str, role: str, 
                      additional_claims: Optional[Dict] = None) -> str:
        """
        Generate a JWT token for a user
        
        Args:
            user_id: Unique user identifier
            username: Username
            role: User role
            additional_claims: Additional claims to include in token
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)  # JWT ID for token tracking
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Log token generation (in production, store in database)
        logger.info(f"Token generated for user {username} (role: {role})")
        
        return token
    
    def decode_token(self, token: str) -> Dict:
        """
        Decode and validate a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def validate_api_key(self, api_key: str) -> Optional[str]:
        """
        Validate an API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Service name if valid, None otherwise
        """
        return self.api_keys.get(api_key)
    
    def check_permission(self, role: str, permission: str) -> bool:
        """
        Check if a role has a specific permission
        
        Args:
            role: User role
            permission: Permission to check
            
        Returns:
            True if role has permission, False otherwise
        """
        if role not in self.roles:
            return False
        
        role_data = self.roles[role]
        
        # Admin has all permissions
        if '*' in role_data['permissions']:
            return True
        
        # Check specific permission
        return permission in role_data['permissions']
    
    def get_role_level(self, role: str) -> int:
        """
        Get the privilege level of a role
        
        Args:
            role: User role
            
        Returns:
            Role level (higher = more privileges)
        """
        return self.roles.get(role, {}).get('level', 0)


# Global auth manager instance
auth_manager = AuthManager()


def require_auth(f):
    """
    Decorator to require authentication for an endpoint
    
    Usage:
        @app.route('/api/protected')
        @require_auth
        def protected_endpoint():
            # Access current user via request.current_user
            return jsonify({"user": request.current_user})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            # Check for API key as fallback
            api_key = request.headers.get('X-API-Key')
            if api_key:
                service = auth_manager.validate_api_key(api_key)
                if service:
                    # Set service context
                    request.current_user = {
                        'user_id': f'service_{service}',
                        'username': service,
                        'role': 'service',
                        'is_service': True
                    }
                    return f(*args, **kwargs)
            
            return jsonify({"error": "Authentication required"}), 401
        
        # Extract token from header
        try:
            parts = auth_header.split()
            if parts[0].lower() != 'bearer' or len(parts) != 2:
                return jsonify({"error": "Invalid authorization header format"}), 401
            
            token = parts[1]
            
            # Decode and validate token
            payload = auth_manager.decode_token(token)
            
            # Set current user in request context
            request.current_user = payload
            
            return f(*args, **kwargs)
            
        except AuthenticationError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({"error": "Authentication failed"}), 401
    
    return decorated_function


def require_permission(permission: str):
    """
    Decorator to require specific permission for an endpoint
    
    Args:
        permission: Required permission string
        
    Usage:
        @app.route('/api/inventory/update')
        @require_auth
        @require_permission('inventory.write')
        def update_inventory():
            return jsonify({"status": "updated"})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Ensure user is authenticated first
            if not hasattr(request, 'current_user'):
                return jsonify({"error": "Authentication required"}), 401
            
            # Service accounts have all permissions
            if request.current_user.get('is_service'):
                return f(*args, **kwargs)
            
            # Check user permission
            user_role = request.current_user.get('role')
            if not auth_manager.check_permission(user_role, permission):
                return jsonify({
                    "error": "Insufficient permissions",
                    "required": permission,
                    "user_role": user_role
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_role(minimum_role: str):
    """
    Decorator to require minimum role level for an endpoint
    
    Args:
        minimum_role: Minimum required role
        
    Usage:
        @app.route('/api/admin/users')
        @require_auth
        @require_role('manager')
        def manage_users():
            return jsonify({"users": []})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Ensure user is authenticated first
            if not hasattr(request, 'current_user'):
                return jsonify({"error": "Authentication required"}), 401
            
            # Service accounts have admin level
            if request.current_user.get('is_service'):
                return f(*args, **kwargs)
            
            # Check role level
            user_role = request.current_user.get('role')
            user_level = auth_manager.get_role_level(user_role)
            required_level = auth_manager.get_role_level(minimum_role)
            
            if user_level < required_level:
                return jsonify({
                    "error": "Insufficient role level",
                    "required_role": minimum_role,
                    "user_role": user_role
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


class SessionManager:
    """Manage user sessions and token blacklisting"""
    
    def __init__(self):
        # In production, use Redis or database
        self.active_sessions = {}
        self.blacklisted_tokens = set()
    
    def create_session(self, user_id: str, token: str, metadata: Optional[Dict] = None):
        """Create a new session"""
        session_data = {
            'token': token,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = []
        
        self.active_sessions[user_id].append(session_data)
    
    def invalidate_session(self, user_id: str, token: Optional[str] = None):
        """Invalidate a user session"""
        if token:
            self.blacklisted_tokens.add(token)
        
        if user_id in self.active_sessions:
            if token:
                # Remove specific session
                self.active_sessions[user_id] = [
                    s for s in self.active_sessions[user_id] 
                    if s['token'] != token
                ]
            else:
                # Remove all sessions for user
                for session in self.active_sessions[user_id]:
                    self.blacklisted_tokens.add(session['token'])
                del self.active_sessions[user_id]
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted"""
        return token in self.blacklisted_tokens
    
    def update_activity(self, user_id: str, token: str):
        """Update last activity time for a session"""
        if user_id in self.active_sessions:
            for session in self.active_sessions[user_id]:
                if session['token'] == token:
                    session['last_activity'] = datetime.utcnow()
                    break
    
    def cleanup_expired_sessions(self, max_inactive_hours: int = 24):
        """Remove expired sessions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_inactive_hours)
        
        for user_id in list(self.active_sessions.keys()):
            active = []
            for session in self.active_sessions[user_id]:
                if session['last_activity'] > cutoff_time:
                    active.append(session)
                else:
                    self.blacklisted_tokens.add(session['token'])
            
            if active:
                self.active_sessions[user_id] = active
            else:
                del self.active_sessions[user_id]


# Global session manager instance
session_manager = SessionManager()