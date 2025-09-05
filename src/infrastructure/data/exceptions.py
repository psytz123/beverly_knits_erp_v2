"""Custom exceptions for data layer operations."""


class DataException(Exception):
    """Base exception for all data-related errors."""
    pass


class DataSourceException(DataException):
    """Exception raised when a data source is unavailable or fails."""
    pass


class DataLoadException(DataException):
    """Exception raised when data loading fails."""
    pass


class DataValidationException(DataException):
    """Exception raised when data validation fails."""
    pass


class DataTransformException(DataException):
    """Exception raised during data transformation."""
    pass


class CacheException(DataException):
    """Exception raised for cache-related errors."""
    pass


class ColumnMappingException(DataException):
    """Exception raised for column mapping errors."""
    pass


class DataIntegrityException(DataException):
    """Exception raised when data integrity checks fail."""
    pass