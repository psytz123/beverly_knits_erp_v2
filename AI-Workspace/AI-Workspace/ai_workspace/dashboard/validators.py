"""Request validation models for API endpoints.

Provides Pydantic models for validating query parameters, path parameters,
and request bodies across all API endpoints.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


# Base query parameter models
class QueryParams(BaseModel):
    """Base query parameter validation with pagination support.

    Attributes:
        limit: Maximum number of items to return (1-100)
        offset: Number of items to skip (0+)
    """

    limit: Optional[int] = Field(50, ge=1, le=100, description="Maximum number of results")
    offset: Optional[int] = Field(0, ge=0, description="Number of results to skip")

    class Config:
        """Pydantic configuration."""

        str_strip_whitespace = True
        validate_assignment = True


class DateRangeParams(QueryParams):
    """Query parameters with date range filtering.

    Attributes:
        start_date: Start date in ISO 8601 format (YYYY-MM-DD)
        end_date: End date in ISO 8601 format (YYYY-MM-DD)
    """

    start_date: Optional[str] = Field(None, description="Start date (ISO 8601)")
    end_date: Optional[str] = Field(None, description="End date (ISO 8601)")

    @validator("start_date", "end_date")
    def validate_iso_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate date is in ISO 8601 format.

        Args:
            v: Date string to validate

        Returns:
            Validated date string

        Raises:
            ValueError: If date format is invalid
        """
        if v is None:
            return v

        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO 8601 (YYYY-MM-DD)")

    @validator("end_date")
    def validate_date_range(cls, v: Optional[str], values: dict) -> Optional[str]:
        """Validate end_date is after start_date.

        Args:
            v: End date string
            values: Dictionary containing validated start_date

        Returns:
            Validated end date string

        Raises:
            ValueError: If end_date is before start_date
        """
        if v is None or values.get("start_date") is None:
            return v

        start = datetime.fromisoformat(values["start_date"])
        end = datetime.fromisoformat(v)

        if end < start:
            raise ValueError("end_date must be after start_date")

        return v


# Phase-specific validators
class PhaseValidator(BaseModel):
    """Validator for phase gate phase names.

    Attributes:
        phase: Phase name (must be valid phase)
    """

    phase: Literal["discovery", "design", "implementation", "verification", "integration"]

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class TaskNameValidator(BaseModel):
    """Validator for task names in query parameters.

    Attributes:
        task: Optional task name filter
    """

    task: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Task name filter"
    )

    @validator("task")
    def validate_task_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate task name format.

        Args:
            v: Task name to validate

        Returns:
            Validated task name

        Raises:
            ValueError: If task name contains invalid characters
        """
        if v is None:
            return v

        # Allow alphanumeric, hyphens, underscores, and spaces
        if not all(c.isalnum() or c in "-_ " for c in v):
            raise ValueError(
                "Task name can only contain alphanumeric characters, hyphens, underscores, and spaces"
            )

        return v.strip()


# Metrics endpoint validators
class MetricsSummaryParams(QueryParams):
    """Query parameters for metrics summary endpoint.

    No additional parameters beyond base pagination.
    """

    pass


class ReuseMetricsParams(QueryParams):
    """Query parameters for reuse metrics endpoint.

    No additional parameters beyond base pagination.
    """

    pass


class TrendsParams(DateRangeParams):
    """Query parameters for trends endpoint.

    Extends DateRangeParams with trend-specific options.
    """

    pass


# Gates endpoint validators
class GateStatusParams(BaseModel):
    """Query parameters for gate status endpoint.

    No query parameters required.
    """

    pass


class PhaseDetailsParams(TaskNameValidator):
    """Query parameters for phase details endpoint.

    Attributes:
        task: Optional task name filter
    """

    pass


class ListGatesParams(TaskNameValidator):
    """Query parameters for list gates endpoint.

    Attributes:
        task: Optional task name filter
    """

    pass


# Reuse endpoint validators
class ReuseChecksParams(QueryParams):
    """Query parameters for reuse checks endpoint.

    Inherits pagination from QueryParams.
    """

    pass


class ReuseHistoryParams(DateRangeParams):
    """Query parameters for reuse history endpoint.

    Inherits date range filtering from DateRangeParams.
    """

    pass


class ReuseViolationsParams(QueryParams):
    """Query parameters for reuse violations endpoint.

    Attributes:
        severity: Optional severity filter
    """

    severity: Optional[Literal["high", "medium", "low"]] = Field(
        None,
        description="Filter by violation severity"
    )


# Stack endpoint validators
class StackDetectParams(BaseModel):
    """Query parameters for stack detection endpoint.

    No query parameters required.
    """

    pass


class RecommendedAgentsParams(BaseModel):
    """Query parameters for recommended agents endpoint.

    No query parameters required.
    """

    pass


class LanguagesParams(QueryParams):
    """Query parameters for languages endpoint.

    Attributes:
        min_confidence: Minimum confidence score (0-100)
    """

    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Minimum confidence score"
    )


class ProjectInfoParams(BaseModel):
    """Query parameters for project info endpoint.

    No query parameters required.
    """

    pass


# Validation helper functions
def validate_query_params(model_class: type[BaseModel], args: dict) -> BaseModel:
    """Validate query parameters against a Pydantic model.

    Args:
        model_class: Pydantic model class to validate against
        args: Dictionary of query parameters from request.args

    Returns:
        Validated model instance

    Raises:
        ValidationError: If validation fails
    """
    return model_class(**args)


def get_validated_params(model_class: type[BaseModel], args: dict) -> dict:
    """Validate and return parameters as dictionary.

    Args:
        model_class: Pydantic model class to validate against
        args: Dictionary of query parameters from request.args

    Returns:
        Validated parameters as dictionary

    Raises:
        ValidationError: If validation fails
    """
    validated = validate_query_params(model_class, args)
    return validated.dict(exclude_none=True)
