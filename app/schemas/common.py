"""
Common Pydantic Schemas

This module demonstrates:
- Reusable base schemas
- Pagination patterns
- Standard API response wrappers
- Custom validators and field types
"""

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Generic type for paginated data
T = TypeVar("T")


class BaseSchema(BaseModel):
    """
    Base schema with common configuration.

    All schemas should inherit from this to ensure consistent behavior.
    """

    model_config = ConfigDict(
        from_attributes=True,  # Allow creation from ORM models
        populate_by_name=True,  # Allow using field names or aliases
        str_strip_whitespace=True,  # Strip whitespace from strings
    )


class TimestampMixin(BaseModel):
    """Mixin for schemas that include timestamps."""

    created_at: datetime
    updated_at: datetime | None = None


class PaginationParams(BaseModel):
    """Query parameters for paginated endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=20, ge=1, le=100, description="Number of items per page"
    )

    @property
    def offset(self) -> int:
        """Calculate SQL offset from page number."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response wrapper.

    Usage:
        PaginatedResponse[AssetResponse]
    """

    items: list[T]
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    pages: int = Field(description="Total number of pages")

    @classmethod
    def create(
        cls, items: list[T], total: int, page: int, page_size: int
    ) -> "PaginatedResponse[T]":
        """Create a paginated response with calculated page count."""
        pages = (total + page_size - 1) // page_size  # Ceiling division
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = Field(default=None, description="Field that caused the error")
    message: str = Field(description="Error message")
    code: str | None = Field(default=None, description="Error code for programmatic handling")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type/title")
    message: str = Field(description="Human-readable error message")
    details: list[ErrorDetail] | None = Field(
        default=None, description="Additional error details"
    )
    request_id: str | None = Field(
        default=None, description="Request ID for support/debugging"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": "email",
                            "message": "Invalid email format",
                            "code": "invalid_format",
                        }
                    ],
                    "request_id": "req_abc123",
                }
            ]
        }
    }


class SuccessResponse(BaseModel):
    """Generic success response for operations that don't return data."""

    success: bool = True
    message: str = Field(description="Success message")