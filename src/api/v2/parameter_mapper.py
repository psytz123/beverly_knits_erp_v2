#!/usr/bin/env python3
"""
Parameter Mapping Module - Phase 4
Maps old API parameters to new v2 parameter structure
"""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ParameterMapper:
    """Maps legacy API parameters to v2 consolidated endpoint parameters"""
    
    # Define parameter mappings for each deprecated endpoint
    PARAMETER_MAPPINGS = {
        # Inventory endpoints mappings
        'yarn-inventory': {
            'include_shortages': 'shortage_only',
            'show_all': 'view=detailed',
            'format_type': 'format',
            'real_time': 'realtime'
        },
        
        'inventory-intelligence-enhanced': {
            'intelligence_level': 'analysis',
            'include_ml': 'analysis=forecast',
            'shortage_analysis': 'analysis=shortage'
        },
        
        # Production endpoints mappings
        'production-planning': {
            'include_ml': 'include_forecast',
            'machine': 'machine_id',
            'order_status': 'status',
            'show_unassigned': 'status=unassigned'
        },
        
        'machine-assignment-suggestions': {
            'work_center': 'machine_id',
            'show_all': 'view=detailed',
            'optimization_level': 'view=recommendations'
        },
        
        # Forecasting endpoints mappings
        'ml-forecasting': {
            'forecast_days': 'horizon',
            'model_type': 'model',
            'include_accuracy': 'detail=accuracy',
            'style': 'style_id'
        },
        
        'sales-forecasting': {
            'period': 'horizon',
            'product': 'style_id',
            'show_confidence': 'detail=full'
        },
        
        # Analytics endpoints mappings
        'comprehensive-kpis': {
            'kpi_type': 'category',
            'real_time': 'realtime',
            'time_period': 'period'
        },
        
        # Yarn management mappings
        'yarn-intelligence': {
            'yarn': 'yarn_id',
            'show_substitutes': 'include_substitutes',
            'intelligence_type': 'action'
        },
        
        'yarn-substitution-intelligent': {
            'yarn': 'yarn_id',
            'criteria_type': 'action=substitution'
        }
    }
    
    # Value transformations for specific parameters
    VALUE_TRANSFORMATIONS = {
        'intelligence_level': {
            'basic': 'none',
            'advanced': 'intelligence',
            'full': 'intelligence'
        },
        'model_type': {
            'time_series': 'arima',
            'neural': 'lstm',
            'boosting': 'xgboost',
            'all': 'ensemble'
        },
        'kpi_type': {
            'operational': 'performance',
            'financial': 'business',
            'all_kpis': 'all'
        }
    }
    
    @classmethod
    def map_parameters(cls, endpoint_name: str, old_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map old endpoint parameters to new v2 structure
        
        Args:
            endpoint_name: Name of the deprecated endpoint (without /api/ prefix)
            old_params: Dictionary of old parameters
            
        Returns:
            Dictionary of mapped v2 parameters
        """
        # Clean endpoint name
        endpoint_name = endpoint_name.replace('/api/', '').replace('/', '')
        
        # Get mapping for this endpoint
        mapping = cls.PARAMETER_MAPPINGS.get(endpoint_name, {})
        
        # Create new parameters dictionary
        new_params = {}
        
        for old_key, old_value in old_params.items():
            # Check if this parameter needs mapping
            if old_key in mapping:
                new_key = mapping[old_key]
                
                # Handle special cases where value contains both key and value
                if '=' in str(new_key):
                    # This is a fixed parameter assignment
                    key, value = new_key.split('=', 1)
                    new_params[key] = value
                else:
                    # Regular parameter mapping
                    new_params[new_key] = cls._transform_value(old_key, old_value)
            else:
                # Keep parameter as-is if no mapping defined
                new_params[old_key] = old_value
        
        return new_params
    
    @classmethod
    def _transform_value(cls, param_name: str, value: Any) -> Any:
        """
        Transform parameter values if needed
        
        Args:
            param_name: Parameter name
            value: Original value
            
        Returns:
            Transformed value
        """
        if param_name in cls.VALUE_TRANSFORMATIONS:
            transformations = cls.VALUE_TRANSFORMATIONS[param_name]
            if str(value) in transformations:
                return transformations[str(value)]
        
        # Handle boolean conversions
        if isinstance(value, bool):
            return 'true' if value else 'false'
        
        # Handle None values
        if value is None:
            return ''
        
        return value
    
    @classmethod
    def extract_endpoint_params(cls, full_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract endpoint name and parameters from full path
        
        Args:
            full_path: Full endpoint path with query parameters
            
        Returns:
            Tuple of (endpoint_name, parameters_dict)
        """
        # Split path and query string
        if '?' in full_path:
            path, query_string = full_path.split('?', 1)
            
            # Parse query parameters
            params = {}
            for param_pair in query_string.split('&'):
                if '=' in param_pair:
                    key, value = param_pair.split('=', 1)
                    params[key] = value
                else:
                    params[param_pair] = 'true'
        else:
            path = full_path
            params = {}
        
        # Clean path to get endpoint name
        endpoint_name = path.replace('/api/', '').strip('/')
        
        return endpoint_name, params
    
    @classmethod
    def get_v2_url(cls, deprecated_url: str) -> str:
        """
        Convert deprecated URL to v2 URL with proper parameter mapping
        
        Args:
            deprecated_url: Full deprecated endpoint URL
            
        Returns:
            v2 endpoint URL with mapped parameters
        """
        from src.api.v2.blueprint_integration import DEPRECATED_ENDPOINT_MAPPINGS
        
        # Extract path and parameters
        if '?' in deprecated_url:
            base_path, _ = deprecated_url.split('?', 1)
        else:
            base_path = deprecated_url
        
        # Get v2 endpoint from mapping
        if base_path not in DEPRECATED_ENDPOINT_MAPPINGS:
            # Not a deprecated endpoint
            return deprecated_url
        
        v2_base = DEPRECATED_ENDPOINT_MAPPINGS[base_path]
        
        # Extract and map parameters
        endpoint_name, old_params = cls.extract_endpoint_params(deprecated_url)
        new_params = cls.map_parameters(endpoint_name, old_params)
        
        # Build v2 URL
        if '?' in v2_base:
            # v2 base already has parameters
            base, existing_params_str = v2_base.split('?', 1)
            
            # Parse existing parameters
            existing_params = {}
            for param in existing_params_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    existing_params[key] = value
            
            # Merge parameters (new params override existing)
            all_params = {**existing_params, **new_params}
        else:
            base = v2_base
            all_params = new_params
        
        # Build final URL
        if all_params:
            param_str = '&'.join([f"{k}={v}" for k, v in all_params.items()])
            return f"{base}?{param_str}"
        
        return base
    
    @classmethod
    def log_parameter_mapping(cls, old_url: str, new_url: str):
        """Log parameter mapping for debugging and analytics"""
        logger.info(f"Parameter mapping: {old_url} -> {new_url}")


class RequestTransformer:
    """Transform request bodies between old and new API formats"""
    
    @staticmethod
    def transform_request_body(endpoint: str, old_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform request body from old format to v2 format
        
        Args:
            endpoint: Endpoint name
            old_body: Original request body
            
        Returns:
            Transformed request body for v2
        """
        # Define body transformations per endpoint
        transformations = {
            'production-order': {
                'order_id': 'id',
                'product': 'style_id',
                'quantity': 'quantity',
                'due_date': 'deadline',
                'priority': 'priority'
            },
            'yarn-update': {
                'yarn_code': 'yarn_id',
                'new_balance': 'balance',
                'adjustment_reason': 'reason'
            }
        }
        
        # Get transformation for this endpoint
        if endpoint in transformations:
            mapping = transformations[endpoint]
            new_body = {}
            
            for old_key, old_value in old_body.items():
                new_key = mapping.get(old_key, old_key)
                new_body[new_key] = old_value
            
            return new_body
        
        # Return unchanged if no transformation defined
        return old_body


class ResponseTransformer:
    """Transform responses from v2 format to legacy format for backward compatibility"""
    
    @staticmethod
    def transform_response(endpoint: str, v2_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform v2 response to legacy format
        
        Args:
            endpoint: Original endpoint name
            v2_response: Response from v2 endpoint
            
        Returns:
            Response in legacy format
        """
        # Extract data from standardized response
        if 'data' in v2_response:
            data = v2_response['data']
        else:
            data = v2_response
        
        # Define response transformations per endpoint
        transformations = {
            'yarn-inventory': lambda d: {
                'yarns': d.get('items', []),
                'total_count': len(d.get('items', [])),
                'shortage_count': len([i for i in d.get('items', []) if i.get('planning_balance', 0) < 0])
            },
            'production-status': lambda d: {
                'orders': d.get('orders', []),
                'summary': {
                    'total': len(d.get('orders', [])),
                    'assigned': len([o for o in d.get('orders', []) if o.get('machine_id')]),
                    'unassigned': len([o for o in d.get('orders', []) if not o.get('machine_id')])
                }
            },
            'comprehensive-kpis': lambda d: {
                'kpis': d.get('kpis', {}),
                'timestamp': d.get('timestamp'),
                'status': 'success'
            }
        }
        
        # Apply transformation if defined
        endpoint_clean = endpoint.replace('/api/', '').strip('/')
        if endpoint_clean in transformations:
            return transformations[endpoint_clean](data)
        
        # Return data as-is if no transformation defined
        return data


# Export main classes
__all__ = [
    'ParameterMapper',
    'RequestTransformer',
    'ResponseTransformer'
]