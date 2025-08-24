#!/usr/bin/env python3
"""
Input Validation Module for Beverly Knits ERP
Provides comprehensive input validation for all API endpoints
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import pandas as pd


class InputValidator:
    """Comprehensive input validation for ERP system"""
    
    # Regex patterns for common validations
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^\+?1?\d{9,15}$',
        'alphanumeric': r'^[a-zA-Z0-9]+$',
        'yarn_id': r'^[A-Z0-9\-]+$',  # Yarn IDs are uppercase alphanumeric with dashes
        'style_id': r'^[A-Z0-9\-\.]+$',  # Style IDs can have dots
        'order_id': r'^[A-Z0-9\-]+$',
        'color_code': r'^#[0-9A-Fa-f]{6}$',
        'percentage': r'^(100|[0-9]{1,2})(\.[0-9]{1,2})?$'
    }
    
    @staticmethod
    def validate_yarn_id(yarn_id: str) -> bool:
        """
        Validate yarn ID format
        
        Args:
            yarn_id: Yarn identifier to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not yarn_id or not isinstance(yarn_id, str):
            return False
        
        # Check pattern and length
        if not re.match(InputValidator.PATTERNS['yarn_id'], yarn_id):
            return False
        
        # Yarn IDs should be between 3 and 20 characters
        if len(yarn_id) < 3 or len(yarn_id) > 20:
            return False
        
        return True
    
    @staticmethod
    def validate_quantity(quantity: Union[int, float, str], 
                         min_value: float = 0, 
                         max_value: Optional[float] = None,
                         allow_negative: bool = False) -> Optional[float]:
        """
        Validate and convert quantity value
        
        Args:
            quantity: Quantity to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_negative: Whether to allow negative values
            
        Returns:
            Validated quantity as float, or None if invalid
        """
        try:
            qty = float(quantity)
            
            if not allow_negative and qty < 0:
                return None
            
            if qty < min_value:
                return None
            
            if max_value is not None and qty > max_value:
                return None
            
            return qty
            
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str, 
                           date_format: str = "%Y-%m-%d",
                           max_range_days: Optional[int] = None) -> bool:
        """
        Validate a date range
        
        Args:
            start_date: Start date string
            end_date: End date string
            date_format: Expected date format
            max_range_days: Maximum allowed range in days
            
        Returns:
            True if valid, False otherwise
        """
        try:
            start = datetime.strptime(start_date, date_format)
            end = datetime.strptime(end_date, date_format)
            
            # Start should be before or equal to end
            if start > end:
                return False
            
            # Check maximum range if specified
            if max_range_days:
                delta = (end - start).days
                if delta > max_range_days:
                    return False
            
            # Don't allow dates too far in the future (e.g., > 5 years)
            future_limit = datetime.now().replace(year=datetime.now().year + 5)
            if end > future_limit:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_price(price: Union[float, str], 
                      currency: str = "USD",
                      min_price: float = 0,
                      max_price: float = 1000000) -> Optional[float]:
        """
        Validate price value
        
        Args:
            price: Price to validate
            currency: Currency code
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            
        Returns:
            Validated price as float, or None if invalid
        """
        try:
            # Remove currency symbols if present
            if isinstance(price, str):
                price = price.replace('$', '').replace(',', '').strip()
            
            price_value = float(price)
            
            # Price should be non-negative
            if price_value < min_price:
                return None
            
            # Check maximum price (sanity check)
            if price_value > max_price:
                return None
            
            # Round to 2 decimal places for currency
            return round(price_value, 2)
            
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def validate_percentage(value: Union[float, str]) -> Optional[float]:
        """
        Validate percentage value (0-100)
        
        Args:
            value: Percentage value to validate
            
        Returns:
            Validated percentage as float, or None if invalid
        """
        try:
            # Handle string percentages like "50%" or "50.5"
            if isinstance(value, str):
                value = value.replace('%', '').strip()
            
            pct = float(value)
            
            # Percentage should be between 0 and 100
            if pct < 0 or pct > 100:
                return None
            
            return round(pct, 2)
            
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False
        
        return bool(re.match(InputValidator.PATTERNS['email'], email.lower()))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not phone or not isinstance(phone, str):
            return False
        
        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        
        return bool(re.match(InputValidator.PATTERNS['phone'], cleaned))
    
    @staticmethod
    def validate_color_code(color: str) -> bool:
        """
        Validate color code (hex format)
        
        Args:
            color: Color code to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not color or not isinstance(color, str):
            return False
        
        return bool(re.match(InputValidator.PATTERNS['color_code'], color))
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 255, 
                       allow_special: bool = False) -> str:
        """
        Sanitize string input
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allow_special: Whether to allow special characters
            
        Returns:
            Sanitized string
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text).strip()
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        # Remove or escape special characters if not allowed
        if not allow_special:
            # Keep only alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\-\_\.\,]', '', text)
        
        # Truncate to maximum length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def validate_batch_size(size: int, min_size: int = 1, 
                          max_size: int = 10000) -> bool:
        """
        Validate batch size for bulk operations
        
        Args:
            size: Batch size to validate
            min_size: Minimum batch size
            max_size: Maximum batch size
            
        Returns:
            True if valid, False otherwise
        """
        try:
            batch_size = int(size)
            return min_size <= batch_size <= max_size
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_file_upload(file_data: Dict[str, Any], 
                           allowed_extensions: List[str] = None,
                           max_size_mb: float = 10) -> Dict[str, Any]:
        """
        Validate file upload parameters
        
        Args:
            file_data: File upload data
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        # Default allowed extensions for ERP data files
        if allowed_extensions is None:
            allowed_extensions = ['.xlsx', '.xls', '.csv', '.json']
        
        # Check file name
        if 'filename' not in file_data:
            result["valid"] = False
            result["errors"].append("Filename is required")
            return result
        
        filename = file_data['filename']
        
        # Check extension
        ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in allowed_extensions:
            result["valid"] = False
            result["errors"].append(f"File type {ext} not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        # Check file size if provided
        if 'size' in file_data:
            size_mb = file_data['size'] / (1024 * 1024)
            if size_mb > max_size_mb:
                result["valid"] = False
                result["errors"].append(f"File size {size_mb:.2f}MB exceeds maximum {max_size_mb}MB")
        
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        if safe_filename != filename:
            result["warnings"] = result.get("warnings", [])
            result["warnings"].append(f"Filename sanitized: {safe_filename}")
            result["safe_filename"] = safe_filename
        
        return result
    
    @staticmethod
    def validate_dataframe_columns(df: pd.DataFrame, 
                                  required_columns: List[str],
                                  optional_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate DataFrame has required columns
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            optional_columns: List of optional column names
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "missing_required": [],
            "missing_optional": [],
            "extra_columns": []
        }
        
        if df is None or df.empty:
            result["valid"] = False
            result["error"] = "DataFrame is empty or None"
            return result
        
        df_columns = set(df.columns)
        required_set = set(required_columns)
        optional_set = set(optional_columns) if optional_columns else set()
        
        # Check required columns
        missing_required = required_set - df_columns
        if missing_required:
            result["valid"] = False
            result["missing_required"] = list(missing_required)
        
        # Check optional columns
        missing_optional = optional_set - df_columns
        if missing_optional:
            result["missing_optional"] = list(missing_optional)
        
        # Check for extra columns
        expected = required_set | optional_set
        extra = df_columns - expected
        if extra:
            result["extra_columns"] = list(extra)
        
        return result


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal and other issues
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory components
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    
    # Remove special characters except dots and underscores
    filename = re.sub(r'[^a-zA-Z0-9\.\-\_]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 250 - len(ext) - 1 if ext else 255
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename


class RequestValidator:
    """Validate incoming API requests"""
    
    @staticmethod
    def validate_pagination(request_args: Dict[str, Any]) -> Dict[str, int]:
        """
        Validate pagination parameters
        
        Args:
            request_args: Request arguments
            
        Returns:
            Dictionary with validated page and per_page values
        """
        page = request_args.get('page', 1)
        per_page = request_args.get('per_page', 50)
        
        try:
            page = max(1, int(page))
            per_page = max(1, min(500, int(per_page)))  # Max 500 items per page
        except (ValueError, TypeError):
            page = 1
            per_page = 50
        
        return {"page": page, "per_page": per_page}
    
    @staticmethod
    def validate_sort_params(request_args: Dict[str, Any], 
                           allowed_fields: List[str]) -> Dict[str, str]:
        """
        Validate sorting parameters
        
        Args:
            request_args: Request arguments
            allowed_fields: List of fields that can be sorted
            
        Returns:
            Dictionary with validated sort_by and order values
        """
        sort_by = request_args.get('sort_by', allowed_fields[0] if allowed_fields else 'id')
        order = request_args.get('order', 'asc').lower()
        
        # Validate sort field
        if sort_by not in allowed_fields:
            sort_by = allowed_fields[0] if allowed_fields else 'id'
        
        # Validate order
        if order not in ['asc', 'desc']:
            order = 'asc'
        
        return {"sort_by": sort_by, "order": order}
    
    @staticmethod
    def validate_filter_params(request_args: Dict[str, Any], 
                             allowed_filters: Dict[str, type]) -> Dict[str, Any]:
        """
        Validate filter parameters
        
        Args:
            request_args: Request arguments
            allowed_filters: Dictionary of allowed filter names and their types
            
        Returns:
            Dictionary with validated filters
        """
        filters = {}
        
        for filter_name, filter_type in allowed_filters.items():
            if filter_name in request_args:
                value = request_args[filter_name]
                
                try:
                    if filter_type == bool:
                        filters[filter_name] = value.lower() in ['true', '1', 'yes']
                    elif filter_type == int:
                        filters[filter_name] = int(value)
                    elif filter_type == float:
                        filters[filter_name] = float(value)
                    elif filter_type == str:
                        filters[filter_name] = str(value).strip()
                    elif filter_type == list:
                        # Handle comma-separated values
                        if isinstance(value, str):
                            filters[filter_name] = [v.strip() for v in value.split(',')]
                        else:
                            filters[filter_name] = value
                except (ValueError, TypeError):
                    # Skip invalid filter values
                    pass
        
        return filters