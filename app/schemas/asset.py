"""
Pydantic schemas for Asset operations.

These schemas define the shape of request and response data
for asset-related API endpoints.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AssetStatus(str, Enum):
    """Processing status of an asset."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AssetBase(BaseModel):
    """Base schema with common asset fields."""

    filename: str = Field(..., description="Storage filename")
    original_filename: str = Field(..., description="Original uploaded filename")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")


class AssetCreate(BaseModel):
    """
    Schema for asset creation.

    Note: Most fields come from the uploaded file itself,
    not from request body. This schema is primarily for
    internal use when creating Asset records.
    """

    filename: str
    original_filename: str
    content_type: str
    file_size: int
    storage_path: str
    width: int | None = None
    height: int | None = None


class AssetResponse(BaseModel):
    """Schema for asset responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    width: int | None = None
    height: int | None = None
    status: str
    created_at: datetime
    processed_at: datetime | None = None

    # URL will be computed by the endpoint
    url: str | None = None


class AssetDetail(AssetResponse):
    """Detailed asset response including ML-extracted data."""

    ml_labels: list[str] | None = None
    ml_colors: list[dict] | None = None
    ml_text: str | None = None
    error_message: str | None = None


class AssetList(BaseModel):
    """Paginated list of assets."""

    items: list[AssetResponse]
    total: int
    page: int
    page_size: int
    pages: int


class AssetUploadResponse(BaseModel):
    """Response after successful upload."""

    id: int
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    status: str
    message: str = "File uploaded successfully. Processing will begin shortly."


class AssetProcessingStatus(BaseModel):
    """Status update for asset processing."""

    id: int
    status: str
    progress: int | None = Field(None, ge=0, le=100, description="Processing progress percentage")
    message: str | None = None
    error: str | None = None