"""
Base utilities and shared functionality for v2 API endpoints
"""

from flask import request, jsonify
from functools import wraps
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class APIv2Base:
    """Base class for v2 API endpoints with common functionality"""
    
    @staticmethod
    def parse_parameters(request_args: Dict) -> Dict[str, Any]:
        """
        Parse and validate common parameters from request
        
        Args:
            request_args: Request arguments from Flask
            
        Returns:
            Parsed and validated parameters
        """
        params = {}
        
        # Boolean parameters
        bool_params = ['realtime', 'ai_enhanced', 'include_forecast', 
                      'include_substitutes', 'include_intelligence',
                      'include_recommendations', 'use_cache']
        
        for param in bool_params:
            if param in request_args:
                params[param] = request_args.get(param, '').lower() in ['true', '1', 'yes']
        
        # String parameters
        string_params = ['operation', 'view', 'analysis_type', 'format',
                        'resource', 'model', 'target', 'output', 'scope',
                        'type', 'level', 'severity', 'phase', 'mode',
                        'time_period', 'yarn_id', 'style_id', 'order_id',
                        'work_center_id', 'machine_id']
        
        for param in string_params:
            if param in request_args:
                params[param] = request_args.get(param)
        
        # Integer parameters
        int_params = ['horizon', 'limit', 'offset', 'days']
        
        for param in int_params:
            if param in request_args:
                try:
                    params[param] = int(request_args.get(param))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid integer value for {param}: {request_args.get(param)}")
        
        # Date range parameters
        if 'date_range' in request_args:
            date_range = request_args.get('date_range', '')
            if ',' in date_range:
                dates = date_range.split(',')
                if len(dates) == 2:
                    params['start_date'] = dates[0].strip()
                    params['end_date'] = dates[1].strip()
        
        return params
    
    @staticmethod
    def validate_parameters(params: Dict, schema: Dict) -> tuple:
        """
        Validate parameters against a schema
        
        Args:
            params: Parameters to validate
            schema: Validation schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        if 'required' in schema:
            for required_param in schema['required']:
                if required_param not in params:
                    return False, f"Missing required parameter: {required_param}"
        
        # Check allowed values
        if 'allowed' in schema:
            for param, allowed_values in schema['allowed'].items():
                if param in params and params[param] not in allowed_values:
                    return False, f"Invalid value for {param}. Allowed: {allowed_values}"
        
        # Check parameter types
        if 'types' in schema:
            for param, expected_type in schema['types'].items():
                if param in params:
                    if not isinstance(params[param], expected_type):
                        return False, f"Invalid type for {param}. Expected: {expected_type.__name__}"
        
        return True, None
    
    @staticmethod
    def standard_response(data: Any, 
                         status: str = "success",
                         metadata: Optional[Dict] = None,
                         error: Optional[str] = None) -> Dict:
        """
        Create standardized API response
        
        Args:
            data: Response data
            status: Status code (success, error, warning)
            metadata: Optional metadata
            error: Error message if applicable
            
        Returns:
            Standardized response dictionary
        """
        response = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data
        }
        
        if metadata:
            response["metadata"] = metadata
        
        if error:
            response["error"] = error
            
        # Add request context
        if request:
            response["request"] = {
                "endpoint": request.endpoint,
                "method": request.method,
                "parameters": dict(request.args)
            }
        
        return response
    
    @staticmethod
    def clean_for_json(obj: Any) -> Any:
        """
        Clean data for JSON serialization
        
        Args:
            obj: Object to clean
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict
            return obj.replace({np.nan: None}).to_dict('records')
        elif isinstance(obj, pd.Series):
            # Convert Series to list
            return obj.replace({np.nan: None}).tolist()
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to list
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            # Convert numpy ints
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            # Convert numpy floats
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, dict):
            # Recursively clean dict
            return {k: APIv2Base.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively clean list
            return [APIv2Base.clean_for_json(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    @staticmethod
    def paginate(data: List, limit: int = 100, offset: int = 0) -> Dict:
        """
        Paginate list data
        
        Args:
            data: List to paginate
            limit: Items per page
            offset: Starting position
            
        Returns:
            Paginated response with metadata
        """
        total = len(data)
        paginated_data = data[offset:offset + limit]
        
        return {
            "items": paginated_data,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }


def v2_endpoint(schema: Optional[Dict] = None):
    """
    Decorator for v2 API endpoints with standard processing
    
    Args:
        schema: Optional parameter validation schema
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Parse parameters
                params = APIv2Base.parse_parameters(request.args)
                
                # Validate if schema provided
                if schema:
                    is_valid, error_msg = APIv2Base.validate_parameters(params, schema)
                    if not is_valid:
                        return jsonify(APIv2Base.standard_response(
                            None, 
                            status="error",
                            error=error_msg
                        )), 400
                
                # Add params to kwargs
                kwargs['params'] = params
                
                # Call the actual endpoint function
                result = func(*args, **kwargs)
                
                # Clean for JSON if needed
                if isinstance(result, dict):
                    result = APIv2Base.clean_for_json(result)
                
                # Ensure it's a proper Flask response
                if not isinstance(result, tuple):
                    return jsonify(result)
                else:
                    return result
                    
            except Exception as e:
                logger.error(f"Error in v2 endpoint {func.__name__}: {str(e)}")
                return jsonify(APIv2Base.standard_response(
                    None,
                    status="error",
                    error=str(e)
                )), 500
        
        return wrapper
    return decorator


def require_auth(func):
    """
    Decorator to require authentication for endpoints
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement actual authentication check
        # For now, just pass through
        return func(*args, **kwargs)
    
    return wrapper


def cache_response(ttl: int = 300):
    """
    Decorator to cache endpoint responses
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request
            import json
            import hashlib
            from src.utils.cache_manager import UnifiedCacheManager
            
            # Initialize cache manager
            cache = UnifiedCacheManager()
            
            # Generate cache key from function name and request data
            cache_key = f"{func.__name__}:{request.full_path}"
            if request.method == 'POST':
                # Include body hash for POST requests
                body_hash = hashlib.md5(request.get_data()).hexdigest()
                cache_key += f":{body_hash}"
            
            # Try to get from cache
            try:
                cached_value = cache.get(cache_key)
                if cached_value:
                    # Return cached response
                    return json.loads(cached_value) if isinstance(cached_value, str) else cached_value
            except Exception as e:
                # Log cache error but continue
                print(f"Cache get error: {e}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            try:
                # Convert to JSON string for caching
                cache_value = json.dumps(result) if not isinstance(result, str) else result
                cache.set(cache_key, cache_value, ttl)
            except Exception as e:
                # Log cache error but return result
                print(f"Cache set error: {e}")
            
            return result
        
        return wrapper
    return decorator