"""
JSON Sanitizer - Cleans NaN and Infinity values from data before JSON serialization
"""
import math
import numpy as np
from typing import Any, Dict, List, Union

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization by converting
    NaN and Infinity values to None or appropriate defaults.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, (float, np.float32, np.float64)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif hasattr(obj, 'to_dict'):
        # Handle pandas DataFrames and Series
        return sanitize_for_json(obj.to_dict())
    else:
        return obj

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default if NaN or invalid.
    """
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, returning default if invalid.
    """
    try:
        # First try to convert to float (handles NaN), then to int
        float_val = float(value)
        if math.isnan(float_val) or math.isinf(float_val):
            return default
        return int(float_val)
    except (TypeError, ValueError):
        return default

def clean_dataframe_for_json(df) -> List[Dict]:
    """
    Clean a pandas DataFrame for JSON serialization.
    """
    if df is None or df.empty:
        return []
    
    # Replace NaN values with None
    df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    # Convert to list of dicts
    result = df_clean.to_dict('records')
    
    # Additional sanitization
    return sanitize_for_json(result)