"""
User and API Key Models

This module demonstrates:
- SQLAlchemy 2.0 style model definitions
- Mapped column types with type hints
- Relationships between models
- Index and constraint definitions
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from asset import Asset
    #from tag import UserTag


class User(Base, TimestampMixin):
    """
    User account model.

    Stores user credentials and profile information.
    Users can have multiple API keys for authentication.
    """

    __tablename__ = "users"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Authentication fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,  # Index for fast lookups by email
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile fields
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",  # Delete API keys when user is deleted
        lazy="selectin",  # Eager load by default
    )

    assets: Mapped[list["Asset"]] = relationship(
        "Asset",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",  # Lazy load assets (could be many)
    )
    """
    tags: Mapped[list["UserTag"]] = relationship(
        "UserTag",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )"""

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class APIKey(Base, TimestampMixin):
    """
    API Key model for machine-to-machine authentication.

    API keys are an alternative to JWT tokens, commonly used for:
    - Server-to-server communication
    - CI/CD pipelines
    - Third-party integrations

    The actual key is only shown once at creation time.
    We store a hash for verification.
    """

    __tablename__ = "api_keys"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign key to user
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Key identification
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="User-provided name for this key (e.g., 'Production Server')",
    )

    # The key prefix (first 8 chars) - stored for identification
    # Example: "vv_abc123..." -> prefix is "vv_abc12"
    key_prefix: Mapped[str] = mapped_column(
        String(12),
        nullable=False,
        index=True,
        comment="First characters of the key for identification",
    )

    # Hashed key for verification (never store the actual key!)
    key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Argon2 hash of the full API key",
    )

    # Status and limits
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Rate limiting tier
    rate_limit_tier: Mapped[str] = mapped_column(
        String(20),
        default="standard",
        nullable=False,
        comment="Rate limit tier: 'standard', 'premium', 'unlimited'",
    )

    # Expiration (optional)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this key expires (null = never)",
    )

    # Last usage tracking
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Scopes/permissions (stored as comma-separated string for simplicity)
    # In production, consider a separate table for fine-grained permissions
    scopes: Mapped[str] = mapped_column(
        Text,
        default="read,write",
        nullable=False,
        comment="Comma-separated list of allowed scopes",
    )

    # Relationship back to user
    user: Mapped["User"] = relationship("User", back_populates="api_keys")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_api_keys_user_active", "user_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"

    @property
    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def scope_list(self) -> list[str]:
        """Get scopes as a list."""
        return [s.strip() for s in self.scopes.split(",") if s.strip()]

    def has_scope(self, scope: str) -> bool:
        """Check if this key has a specific scope."""
        return scope in self.scope_list or "*" in self.scope_list