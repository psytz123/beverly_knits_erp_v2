"""
Feature flags configuration for Beverly Knits ERP API consolidation.
Controls the rollout and behavior of API consolidation features.
"""

from datetime import datetime

# Feature flags for API consolidation
FEATURE_FLAGS = {
    # Main consolidation feature - controls whether new consolidated endpoints are active
    "api_consolidation_enabled": True,
    
    # Redirect deprecated APIs to new endpoints
    "redirect_deprecated_apis": True,
    
    # Log usage of deprecated endpoints for monitoring
    "log_deprecated_usage": True,
    
    # Enforce new APIs only (disable deprecated endpoints entirely)
    "enforce_new_apis": False,
    
    # Enable parameter-based views in consolidated endpoints
    "parameter_based_views": True,
    
    # Enable AI-enhanced responses in consolidated endpoints
    "ai_enhanced_responses": True,
    
    # Enable real-time data updates in consolidated endpoints
    "realtime_data_enabled": True,
    
    # Enable caching for consolidated endpoints
    "consolidated_endpoint_caching": True,
    
    # Enable performance monitoring for consolidated endpoints
    "performance_monitoring": True,
    
    # Enable compatibility layer in dashboard
    "dashboard_compatibility_layer": True,
    
    # ===== eFab API Integration Feature Flags =====
    
    # Main eFab API integration control
    "efab_api_enabled": False,  # Master switch for eFab API integration
    
    # eFab API data source priority
    "efab_api_priority": True,  # Use API as primary data source when enabled
    
    # eFab API fallback mechanism
    "efab_fallback_enabled": True,  # Always fall back to files if API fails
    
    # Gradual rollout controls - percentage of requests to route through API
    "efab_rollout_percentage": 0,  # Overall rollout percentage (0-100)
    "efab_rollout_yarn_inventory": 0,  # Yarn inventory endpoint
    "efab_rollout_knit_orders": 0,  # Knit orders endpoint
    "efab_rollout_po_deliveries": 0,  # PO deliveries endpoint
    "efab_rollout_sales_data": 0,  # Sales data endpoint
    "efab_rollout_greige": 0,  # Greige inventory endpoint
    "efab_rollout_finished": 0,  # Finished goods endpoint
    
    # eFab API monitoring
    "efab_metrics_enabled": True,  # Track API performance metrics
    "efab_log_api_calls": True,  # Log all API calls for debugging
    "efab_validate_api_data": True,  # Validate API responses
    "efab_strict_validation": False,  # Fail on validation errors (production = true)
    
    # eFab API circuit breaker
    "efab_circuit_breaker_enabled": True,  # Enable circuit breaker pattern
    "efab_circuit_breaker_threshold": 5,  # Failures before circuit opens
    
    # eFab API testing/development
    "efab_debug_mode": False,  # Enable detailed debug logging
    "efab_mock_api_responses": False,  # Use mock responses for testing
    "efab_api_dry_run": False  # Simulate API calls without making them
}

# API consolidation metadata
API_CONSOLIDATION_META = {
    "version": "1.0",
    "rollout_date": "2025-08-29",
    "deprecation_start": "2025-09-01",
    "deprecation_end": "2025-10-01",
    "total_endpoints_before": 95,
    "total_endpoints_after": 50,
    "expected_reduction": "47%"
}

# Monitoring configuration
MONITORING_CONFIG = {
    "deprecated_call_logging": True,
    "redirect_performance_tracking": True,
    "error_rate_monitoring": True,
    "response_time_tracking": True,
    "usage_analytics": True
}

# Rollback configuration
ROLLBACK_CONFIG = {
    "emergency_rollback_enabled": True,
    "gradual_rollback_enabled": True,
    "rollback_trigger_error_rate": 0.05,  # 5% error rate triggers rollback
    "rollback_trigger_response_time": 5.0  # 5 second response time triggers rollback
}

def get_feature_flag(flag_name, default=False):
    """
    Get the value of a feature flag.
    
    Args:
        flag_name (str): Name of the feature flag
        default: Default value if flag not found
    
    Returns:
        Flag value or default
    """
    return FEATURE_FLAGS.get(flag_name, default)

def is_consolidation_enabled():
    """Check if API consolidation is enabled."""
    return get_feature_flag("api_consolidation_enabled")

def should_redirect_deprecated():
    """Check if deprecated APIs should redirect."""
    return get_feature_flag("redirect_deprecated_apis")

def should_log_deprecated_usage():
    """Check if deprecated API usage should be logged."""
    return get_feature_flag("log_deprecated_usage")

def should_enforce_new_apis():
    """Check if only new APIs should be available."""
    return get_feature_flag("enforce_new_apis")

def enable_consolidation():
    """Enable API consolidation (for deployment)."""
    global FEATURE_FLAGS
    FEATURE_FLAGS["api_consolidation_enabled"] = True
    print(f"[{datetime.now()}] API Consolidation ENABLED")

def disable_consolidation():
    """Disable API consolidation (for rollback)."""
    global FEATURE_FLAGS
    FEATURE_FLAGS["api_consolidation_enabled"] = False
    print(f"[{datetime.now()}] API Consolidation DISABLED")

def emergency_rollback():
    """Perform emergency rollback of all consolidation features."""
    global FEATURE_FLAGS
    FEATURE_FLAGS.update({
        "api_consolidation_enabled": False,
        "redirect_deprecated_apis": False,
        "enforce_new_apis": False
    })
    print(f"[{datetime.now()}] EMERGENCY ROLLBACK ACTIVATED")

# ===== eFab API Integration Functions =====

def is_efab_api_enabled():
    """Check if eFab API integration is enabled."""
    import os
    # Check environment variable first, then feature flag
    env_enabled = os.getenv('EFAB_API_ENABLED', 'false').lower() == 'true'
    flag_enabled = get_feature_flag("efab_api_enabled")
    return env_enabled or flag_enabled

def get_efab_rollout_percentage(endpoint_type=None):
    """
    Get the rollout percentage for eFab API.
    
    Args:
        endpoint_type: Optional specific endpoint type 
                      (yarn_inventory, knit_orders, po_deliveries, etc.)
    
    Returns:
        int: Rollout percentage (0-100)
    """
    if endpoint_type:
        flag_name = f"efab_rollout_{endpoint_type}"
        return get_feature_flag(flag_name, 0)
    return get_feature_flag("efab_rollout_percentage", 0)

def should_use_efab_api(endpoint_type=None):
    """
    Determine if a request should use eFab API based on rollout percentage.
    
    Args:
        endpoint_type: Optional specific endpoint type
    
    Returns:
        bool: True if should use API, False otherwise
    """
    if not is_efab_api_enabled():
        return False
    
    import random
    rollout = get_efab_rollout_percentage(endpoint_type)
    
    if rollout == 0:
        return False
    elif rollout >= 100:
        return True
    else:
        # Random selection based on rollout percentage
        return random.randint(1, 100) <= rollout

def enable_efab_api():
    """Enable eFab API integration."""
    global FEATURE_FLAGS
    FEATURE_FLAGS["efab_api_enabled"] = True
    print(f"[{datetime.now()}] eFab API Integration ENABLED")

def disable_efab_api():
    """Disable eFab API integration."""
    global FEATURE_FLAGS
    FEATURE_FLAGS["efab_api_enabled"] = False
    FEATURE_FLAGS["efab_rollout_percentage"] = 0
    print(f"[{datetime.now()}] eFab API Integration DISABLED")

def set_efab_rollout(percentage, endpoint_type=None):
    """
    Set the rollout percentage for eFab API.
    
    Args:
        percentage: Rollout percentage (0-100)
        endpoint_type: Optional specific endpoint type
    """
    global FEATURE_FLAGS
    percentage = max(0, min(100, percentage))  # Clamp to 0-100
    
    if endpoint_type:
        flag_name = f"efab_rollout_{endpoint_type}"
        FEATURE_FLAGS[flag_name] = percentage
        print(f"[{datetime.now()}] eFab API rollout for {endpoint_type}: {percentage}%")
    else:
        FEATURE_FLAGS["efab_rollout_percentage"] = percentage
        print(f"[{datetime.now()}] eFab API overall rollout: {percentage}%")

def emergency_disable_efab():
    """Emergency disable of eFab API integration."""
    global FEATURE_FLAGS
    FEATURE_FLAGS.update({
        "efab_api_enabled": False,
        "efab_rollout_percentage": 0,
        "efab_rollout_yarn_inventory": 0,
        "efab_rollout_knit_orders": 0,
        "efab_rollout_po_deliveries": 0,
        "efab_rollout_sales_data": 0,
        "efab_rollout_greige": 0,
        "efab_rollout_finished": 0
    })
    print(f"[{datetime.now()}] eFab API EMERGENCY DISABLED")

# Export commonly used functions
__all__ = [
    'FEATURE_FLAGS',
    'API_CONSOLIDATION_META',
    'MONITORING_CONFIG',
    'ROLLBACK_CONFIG',
    'get_feature_flag',
    'is_consolidation_enabled',
    'should_redirect_deprecated',
    'should_log_deprecated_usage',
    'should_enforce_new_apis',
    'enable_consolidation',
    'disable_consolidation',
    'emergency_rollback',
    # eFab API functions
    'is_efab_api_enabled',
    'get_efab_rollout_percentage',
    'should_use_efab_api',
    'enable_efab_api',
    'disable_efab_api',
    'set_efab_rollout',
    'emergency_disable_efab'
]