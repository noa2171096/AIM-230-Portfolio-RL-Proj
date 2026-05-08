"""
Asset model for tracking uploaded files.

Each asset represents an uploaded image with its metadata,
processing status, and optional ML-extracted features.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from user import User


class AssetStatus(str, Enum):
    """Processing status of an asset."""

    PENDING = "pending"  # Uploaded, not yet processed
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Processing finished successfully
    FAILED = "failed"  # Processing failed


class Asset(Base, TimestampMixin):
    """
    Represents an uploaded image asset.

    Assets go through a lifecycle:
    1. PENDING - File uploaded, awaiting processing
    2. PROCESSING - ML pipeline is analyzing the image
    3. COMPLETED - Analysis done, features extracted
    4. FAILED - Processing encountered an error
    """

    __tablename__ = "assets"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Owner relationship
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # File information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # Storage location (relative path or S3 key)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Image dimensions (populated after upload)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20),
        default=AssetStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ML-extracted features (stored as JSON string)
    # Will contain: labels, colors, text (OCR), embedding vector
    ml_labels: Mapped[str | None] = mapped_column(Text)  # JSON array of labels
    ml_colors: Mapped[str | None] = mapped_column(Text)  # JSON array of colors
    ml_text: Mapped[str | None] = mapped_column(Text)  # OCR extracted text
    embedding_vector: Mapped[str | None] = mapped_column(Text)  # JSON array of floats

    # User-defined tags (stored as JSON array)
    custom_tags: Mapped[str | None] = mapped_column(Text)  # JSON array of user tags

    # Relationships
    user: Mapped["User"] = relationship(back_populates="assets")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_assets_user_status", "user_id", "status"),
        Index("ix_assets_user_created", "user_id", "created_at"),
    )

    @property
    def is_image(self) -> bool:
        """Check if the asset is an image based on content type."""
        return self.content_type.startswith("image/")

    @property
    def is_processed(self) -> bool:
        """Check if the asset has been processed."""
        return self.status == AssetStatus.COMPLETED.value

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """Get image dimensions as a tuple."""
        if self.width and self.height:
            return (self.width, self.height)
        return None

    def __repr__(self) -> str:
        return f"<Asset(id={self.id}, filename={self.original_filename}, status={self.status})>"