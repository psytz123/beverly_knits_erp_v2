"""Error handling middleware for API endpoints.

Provides consistent error responses and exception handling across all endpoints.
"""

from typing import Any, Dict, Tuple

from flask import Flask, jsonify
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException


def register_error_handlers(app: Flask) -> None:
    """Register all error handlers with the Flask application.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(400)
    def bad_request(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 400 Bad Request errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 400 status code
        """
        return jsonify({
            "error": "Bad Request",
            "message": str(error) if error else "Invalid request parameters",
            "status": 400,
            "type": "bad_request"
        }), 400

    @app.errorhandler(404)
    def not_found(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 404 Not Found errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 404 status code
        """
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status": 404,
            "type": "not_found"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 405 Method Not Allowed errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 405 status code
        """
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The requested HTTP method is not allowed for this endpoint",
            "status": 405,
            "type": "method_not_allowed"
        }), 405

    @app.errorhandler(422)
    def unprocessable_entity(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 422 Unprocessable Entity errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 422 status code
        """
        return jsonify({
            "error": "Unprocessable Entity",
            "message": str(error) if error else "Request validation failed",
            "status": 422,
            "type": "validation_error"
        }), 422

    @app.errorhandler(429)
    def rate_limit_exceeded(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 429 Too Many Requests errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 429 status code
        """
        return jsonify({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please try again later.",
            "status": 429,
            "type": "rate_limit_error",
            "retry_after": getattr(error, "retry_after", 60)
        }), 429

    @app.errorhandler(500)
    def internal_error(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 500 Internal Server Error.

        Args:
            error: Error object

        Returns:
            JSON error response with 500 status code
        """
        # Log the error here (future enhancement)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status": 500,
            "type": "internal_error"
        }), 500

    @app.errorhandler(503)
    def service_unavailable(error: Any) -> Tuple[Dict[str, Any], int]:
        """Handle 503 Service Unavailable errors.

        Args:
            error: Error object

        Returns:
            JSON error response with 503 status code
        """
        return jsonify({
            "error": "Service Unavailable",
            "message": "Service temporarily unavailable. Please try again later.",
            "status": 503,
            "type": "service_unavailable"
        }), 503

    @app.errorhandler(ValidationError)
    def validation_error(error: ValidationError) -> Tuple[Dict[str, Any], int]:
        """Handle Pydantic validation errors.

        Args:
            error: Pydantic ValidationError instance

        Returns:
            JSON error response with 422 status code
        """
        errors = []
        for err in error.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            errors.append({
                "field": field,
                "message": err["msg"],
                "type": err["type"]
            })

        return jsonify({
            "error": "Validation Error",
            "message": "Request validation failed",
            "status": 422,
            "type": "validation_error",
            "details": errors
        }), 422

    @app.errorhandler(HTTPException)
    def http_exception(error: HTTPException) -> Tuple[Dict[str, Any], int]:
        """Handle general Werkzeug HTTP exceptions.

        Args:
            error: HTTPException instance

        Returns:
            JSON error response with appropriate status code
        """
        return jsonify({
            "error": error.name,
            "message": error.description,
            "status": error.code,
            "type": "http_error"
        }), error.code or 500

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception) -> Tuple[Dict[str, Any], int]:
        """Handle unexpected exceptions.

        Args:
            error: Exception instance

        Returns:
            JSON error response with 500 status code
        """
        # Log the full exception for debugging
        app.logger.error(f"Unexpected error: {type(error).__name__}: {str(error)}", exc_info=True)

        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status": 500,
            "type": "unexpected_error"
        }), 500


class APIError(Exception):
    """Base class for API-specific errors.

    Attributes:
        message: Error message
        status_code: HTTP status code
        error_type: Error type identifier
        details: Optional additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "api_error",
        details: Dict[str, Any] | None = None
    ) -> None:
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code (default: 500)
            error_type: Error type identifier (default: "api_error")
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "error": self.error_type.replace("_", " ").title(),
            "message": self.message,
            "status": self.status_code,
            "type": self.error_type
        }

        if self.details:
            error_dict["details"] = self.details

        return error_dict


class DataNotFoundError(APIError):
    """Raised when requested data is not found.

    Attributes:
        message: Error message
        resource_type: Type of resource not found
        resource_id: Identifier of the resource
    """

    def __init__(self, message: str, resource_type: str = "resource", resource_id: str | None = None) -> None:
        """Initialize data not found error.

        Args:
            message: Error message
            resource_type: Type of resource not found
            resource_id: Identifier of the resource
        """
        details = {"resource_type": resource_type}
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            status_code=404,
            error_type="data_not_found",
            details=details
        )


class ValidationFailedError(APIError):
    """Raised when data validation fails.

    Attributes:
        message: Error message
        validation_errors: List of validation errors
    """

    def __init__(self, message: str, validation_errors: list[Dict[str, str]] | None = None) -> None:
        """Initialize validation failed error.

        Args:
            message: Error message
            validation_errors: List of validation errors
        """
        super().__init__(
            message=message,
            status_code=422,
            error_type="validation_failed",
            details={"errors": validation_errors or []}
        )


class ServiceError(APIError):
    """Raised when an internal service error occurs.

    Attributes:
        message: Error message
        service_name: Name of the service that failed
    """

    def __init__(self, message: str, service_name: str = "unknown") -> None:
        """Initialize service error.

        Args:
            message: Error message
            service_name: Name of the service that failed
        """
        super().__init__(
            message=message,
            status_code=503,
            error_type="service_error",
            details={"service": service_name}
        )


def handle_api_error(error: APIError) -> Tuple[Dict[str, Any], int]:
    """Handle custom API errors.

    Args:
        error: APIError instance

    Returns:
        JSON error response with appropriate status code
    """
    return jsonify(error.to_dict()), error.status_code
