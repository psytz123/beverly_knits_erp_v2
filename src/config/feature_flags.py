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
    "dashboard_compatibility_layer": True
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
    'emergency_rollback'
]