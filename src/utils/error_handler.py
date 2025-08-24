#!/usr/bin/env python3
"""
Comprehensive Error Handling System for Beverly Knits ERP
Provides decorators and utilities for consistent error handling across all endpoints
"""

import logging
import traceback
from functools import wraps
from typing import Any, Dict, Optional, Tuple
from flask import jsonify, request
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)


class ERPException(Exception):
    """Base exception for ERP system"""
    def __init__(self, message: str, code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ERPException):
    """Raised when input validation fails"""
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, code=400, details=details)


class DataNotFoundError(ERPException):
    """Raised when requested data is not found"""
    def __init__(self, resource: str, identifier: Optional[str] = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(message, code=404)


class ServiceUnavailableError(ERPException):
    """Raised when a required service is unavailable"""
    def __init__(self, service: str, reason: Optional[str] = None):
        message = f"Service '{service}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, code=503)


class InsufficientInventoryError(ERPException):
    """Raised when inventory is insufficient for an operation"""
    def __init__(self, item: str, required: float, available: float):
        message = f"Insufficient inventory for {item}: required {required}, available {available}"
        details = {
            "item": item,
            "required": required,
            "available": available,
            "shortage": required - available
        }
        super().__init__(message, code=409, details=details)


def handle_api_errors(func):
    """
    Decorator to handle errors in API endpoints consistently
    
    Usage:
        @app.route('/api/endpoint')
        @handle_api_errors
        def api_endpoint():
            # endpoint logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Log request details
            logger.debug(f"API Request: {request.method} {request.path}")
            
            # Execute the endpoint function
            result = func(*args, **kwargs)
            
            # If result is already a response tuple, return it
            if isinstance(result, tuple):
                return result
            
            # Otherwise, ensure it's properly formatted
            return result
            
        except ERPException as e:
            # Handle custom ERP exceptions
            logger.warning(f"ERP Error in {func.__name__}: {e.message}")
            return jsonify({
                "error": e.message,
                "code": e.code,
                "details": e.details,
                "timestamp": datetime.now().isoformat()
            }), e.code
            
        except ValueError as e:
            # Handle value errors (often from data conversion)
            logger.error(f"Value Error in {func.__name__}: {str(e)}")
            return jsonify({
                "error": "Invalid data format",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
            
        except KeyError as e:
            # Handle missing keys in data
            logger.error(f"Key Error in {func.__name__}: {str(e)}")
            return jsonify({
                "error": "Missing required data",
                "field": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
            
        except FileNotFoundError as e:
            # Handle missing files
            logger.error(f"File Not Found in {func.__name__}: {str(e)}")
            return jsonify({
                "error": "Required file not found",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 404
            
        except PermissionError as e:
            # Handle permission issues
            logger.error(f"Permission Error in {func.__name__}: {str(e)}")
            return jsonify({
                "error": "Permission denied",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 403
            
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            
            # In production, don't expose internal errors
            if request.environ.get('APP_ENV') == 'production':
                return jsonify({
                    "error": "An internal error occurred",
                    "reference": f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "timestamp": datetime.now().isoformat()
                }), 500
            else:
                # In development, provide more details
                return jsonify({
                    "error": "Internal server error",
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc().split('\n'),
                    "timestamp": datetime.now().isoformat()
                }), 500
    
    return wrapper


def validate_required_fields(data: Dict, required_fields: list) -> None:
    """
    Validate that all required fields are present in the data
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Raises:
        ValidationError: If any required field is missing
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field=missing_fields[0]
        )


def validate_numeric_field(data: Dict, field: str, min_value: Optional[float] = None, 
                          max_value: Optional[float] = None) -> float:
    """
    Validate and convert a numeric field
    
    Args:
        data: Dictionary containing the field
        field: Field name to validate
        min_value: Optional minimum value
        max_value: Optional maximum value
        
    Returns:
        The numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    if field not in data:
        raise ValidationError(f"Field '{field}' is required", field=field)
    
    try:
        value = float(data[field])
    except (ValueError, TypeError):
        raise ValidationError(f"Field '{field}' must be numeric", field=field)
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Field '{field}' must be at least {min_value}", 
            field=field
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Field '{field}' must not exceed {max_value}", 
            field=field
        )
    
    return value


def validate_date_field(data: Dict, field: str, date_format: str = "%Y-%m-%d") -> datetime:
    """
    Validate and parse a date field
    
    Args:
        data: Dictionary containing the field
        field: Field name to validate
        date_format: Expected date format
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValidationError: If validation fails
    """
    if field not in data:
        raise ValidationError(f"Field '{field}' is required", field=field)
    
    try:
        return datetime.strptime(data[field], date_format)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Field '{field}' must be a valid date in format {date_format}", 
            field=field
        )


def validate_enum_field(data: Dict, field: str, allowed_values: list) -> str:
    """
    Validate that a field contains one of the allowed values
    
    Args:
        data: Dictionary containing the field
        field: Field name to validate
        allowed_values: List of allowed values
        
    Returns:
        The field value
        
    Raises:
        ValidationError: If validation fails
    """
    if field not in data:
        raise ValidationError(f"Field '{field}' is required", field=field)
    
    value = data[field]
    if value not in allowed_values:
        raise ValidationError(
            f"Field '{field}' must be one of: {', '.join(map(str, allowed_values))}", 
            field=field
        )
    
    return value


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_percentage(value: float, total: float, decimals: int = 2) -> float:
    """
    Calculate percentage safely, handling zero total
    
    Args:
        value: The value to calculate percentage for
        total: The total value
        decimals: Number of decimal places to round to
        
    Returns:
        Percentage value
    """
    if total == 0:
        return 0.0
    return round((value / total) * 100, decimals)


def format_error_response(error: Exception, context: Optional[str] = None) -> Tuple[Dict, int]:
    """
    Format an error into a consistent response structure
    
    Args:
        error: The exception that occurred
        context: Optional context about where the error occurred
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    if isinstance(error, ERPException):
        response = {
            "error": error.message,
            "code": error.code,
            "details": error.details,
            "timestamp": datetime.now().isoformat()
        }
        status_code = error.code
    else:
        response = {
            "error": "An error occurred",
            "message": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.now().isoformat()
        }
        status_code = 500
    
    if context:
        response["context"] = context
    
    return response, status_code


class ErrorLogger:
    """Centralized error logging with categorization"""
    
    def __init__(self, log_file: str = "errors.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("ErrorLogger")
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.logger.error(json.dumps(error_data, indent=2))
    
    def log_validation_error(self, field: str, value: Any, reason: str):
        """Log a validation error"""
        self.logger.warning(f"Validation failed for field '{field}': {reason} (value: {value})")
    
    def log_api_error(self, endpoint: str, method: str, error: Exception):
        """Log an API error"""
        self.logger.error(f"API Error - {method} {endpoint}: {str(error)}")


# Create a global error logger instance
error_logger = ErrorLogger()


# Validation schemas for common data structures
YARN_SCHEMA = {
    "required": ["Desc#", "Planning_Balance"],
    "numeric": ["Planning_Balance", "Allocated", "On_Order", "Consumed"],
    "optional": ["Description", "Supplier", "Cost/Pound"]
}

ORDER_SCHEMA = {
    "required": ["Order_ID", "Product", "Quantity"],
    "numeric": ["Quantity", "Unit_Price"],
    "dates": ["Order_Date", "Due_Date"],
    "enums": {
        "Status": ["Pending", "In_Progress", "Completed", "Cancelled"]
    }
}


def validate_data_schema(data: Dict, schema: Dict) -> Dict:
    """
    Validate data against a schema definition
    
    Args:
        data: Data to validate
        schema: Schema definition
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    validated = {}
    
    # Check required fields
    if "required" in schema:
        validate_required_fields(data, schema["required"])
        for field in schema["required"]:
            validated[field] = data[field]
    
    # Validate numeric fields
    if "numeric" in schema:
        for field in schema["numeric"]:
            if field in data and data[field] is not None:
                validated[field] = validate_numeric_field(data, field)
    
    # Validate date fields
    if "dates" in schema:
        for field in schema["dates"]:
            if field in data and data[field] is not None:
                validated[field] = validate_date_field(data, field)
    
    # Validate enum fields
    if "enums" in schema:
        for field, allowed_values in schema["enums"].items():
            if field in data and data[field] is not None:
                validated[field] = validate_enum_field(data, field, allowed_values)
    
    # Include optional fields
    if "optional" in schema:
        for field in schema["optional"]:
            if field in data:
                validated[field] = data[field]
    
    return validated