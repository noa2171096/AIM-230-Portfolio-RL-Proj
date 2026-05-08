"""
User and API Key Pydantic Schemas

This module demonstrates:
- Request validation schemas (Create, Update)
- Response schemas (hiding sensitive data)
- Field validation with Pydantic
- Schema inheritance patterns
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# =============================================================================
# User Schemas
# =============================================================================


class UserBase(BaseModel):
    """Base schema with common user fields."""

    email: EmailStr = Field(description="User's email address")
    full_name: str | None = Field(default=None, max_length=255, description="User's full name")


class UserCreate(UserBase):
    """
    Schema for user registration.

    Used as request body for POST /auth/register
    """

    password: str = Field(
        min_length=8,
        max_length=100,
        description="Password (min 8 characters)",
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Validate password meets minimum security requirements.

        In production, consider using a library like `password-strength`
        for more comprehensive validation.
        """
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "user@example.com",
                    "password": "SecurePass123",
                    "full_name": "John Doe",
                }
            ]
        }
    }


class UserUpdate(BaseModel):
    """
    Schema for updating user profile.

    All fields are optional - only provided fields are updated.
    """

    full_name: str | None = Field(default=None, max_length=255)
    # Email change might require re-verification in production


class UserResponse(UserBase):
    """
    Schema for user in API responses.

    Note: Never includes password or sensitive fields.
    """

    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserInDB(UserResponse):
    """
    Schema for user with all database fields.

    Used internally, never exposed via API.
    """

    hashed_password: str
    is_superuser: bool
    updated_at: datetime | None


# =============================================================================
# API Key Schemas
# =============================================================================


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""

    name: str = Field(
        min_length=1,
        max_length=100,
        description="A name to identify this API key",
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Days until expiration (null = never expires)",
    )
    scopes: list[str] = Field(
        default=["read", "write"],
        description="Permissions for this key",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Production Server",
                    "expires_in_days": 90,
                    "scopes": ["read", "write"],
                }
            ]
        }
    }


class APIKeyResponse(BaseModel):
    """
    Schema for API key in responses.

    Does NOT include the actual key (only shown once at creation).
    """

    id: int
    name: str
    key_prefix: str = Field(description="First characters of the key for identification")
    is_active: bool
    rate_limit_tier: str
    scopes: list[str]
    expires_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("scopes", mode="before")
    @classmethod
    def parse_scopes(cls, v):
        """Convert comma-separated string to list."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class APIKeyCreated(APIKeyResponse):
    """
    Schema returned when a new API key is created.

    This is the ONLY time the full key is shown.
    The user must save it - we can't retrieve it later.
    """

    key: str = Field(description="The full API key (save this - shown only once!)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "Production Server",
                    "key": "vv_abc123def456ghi789jkl012mno345pq",
                    "key_prefix": "vv_abc123",
                    "is_active": True,
                    "rate_limit_tier": "standard",
                    "scopes": ["read", "write"],
                    "expires_at": "2024-04-15T00:00:00Z",
                    "last_used_at": None,
                    "created_at": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }


# =============================================================================
# Authentication Schemas
# =============================================================================


class LoginRequest(BaseModel):
    """Schema for login request."""

    email: EmailStr
    password: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "user@example.com",
                    "password": "SecurePass123",
                }
            ]
        }
    }


class TokenResponse(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 1800,
                }
            ]
        }
    }