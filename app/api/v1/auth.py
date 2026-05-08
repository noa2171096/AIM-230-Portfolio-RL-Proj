"""
Adapted from VisualVault

Authentication Endpoints

This module demonstrates:
- User registration endpoint
- Login endpoint (JWT tokens)
- API key management endpoints
- Protected route dependencies
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.database import DbSessionDep
from app.models.user import User
from app.schemas.user import (
    APIKeyCreate,
    APIKeyCreated,
    APIKeyResponse,
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.services.auth import AuthService 

router = APIRouter()

# Security scheme for JWT bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Dependencies
# =============================================================================


async def get_auth_service(db: DbSessionDep) -> AuthService:
    """Dependency to get auth service instance."""
    return AuthService(db)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


async def get_current_user(
    db: DbSessionDep,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> User:
    """
    Dependency to get the current authenticated user.

    Supports two authentication methods:
    1. JWT Bearer token in Authorization header
    2. API key in X-API-Key header

    Usage:
        @router.get("/me")
        async def get_me(user: CurrentUserDep):
            return user
    """
    auth_service = AuthService(db)

    # Try JWT token first
    if credentials:
        user_id = auth_service.verify_access_token(credentials.credentials)
        if user_id:
            user = await auth_service.get_user_by_id(user_id)
            if user and user.is_active:
                return user

    # Try API key
    if x_api_key:
        result = await auth_service.verify_api_key(x_api_key)
        if result:
            _, user = result
            return user

    # No valid authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    db: DbSessionDep,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> User | None:
    """
    Optional authentication - returns None if not authenticated.

    Use for endpoints that work differently for authenticated vs anonymous users.
    """
    try:
        return await get_current_user(db, credentials, x_api_key)
    except HTTPException:
        return None


# Type aliases for dependency injection
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentUserOptionalDep = Annotated[User | None, Depends(get_current_user_optional)]


# =============================================================================
# Registration & Login
# =============================================================================


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password.",
)
async def register(
    data: UserCreate,
    auth_service: AuthServiceDep,
) -> UserResponse:
    """
    Register a new user.

    - **email**: Must be a valid email address, unique in the system
    - **password**: Min 8 chars, must include uppercase, lowercase, and digit
    - **full_name**: Optional display name
    """
    try:
        user = await auth_service.create_user(data)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get access token",
    description="Authenticate with email and password to receive a JWT token.",
)
async def login(
    data: LoginRequest,
    auth_service: AuthServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenResponse:
    """
    Authenticate and get an access token.

    The token should be included in the Authorization header as:
    `Authorization: Bearer <token>`
    """
    user = await auth_service.authenticate_user(data.email, data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth_service.create_access_token(user.id)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


# =============================================================================
# User Profile
# =============================================================================


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get the profile of the currently authenticated user.",
)
async def get_me(user: CurrentUserDep) -> UserResponse:
    """Get the current authenticated user's profile."""
    return UserResponse.model_validate(user)


# =============================================================================
# API Keys
# =============================================================================


@router.post(
    "/api-keys",
    response_model=APIKeyCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Generate a new API key for programmatic access.",
)
async def create_api_key(
    data: APIKeyCreate,
    user: CurrentUserDep,
    auth_service: AuthServiceDep,
) -> APIKeyCreated:
    """
    Create a new API key.

    **Important:** The full key is only shown once in this response.
    Store it securely - you won't be able to retrieve it again.
    """
    api_key, plain_key = await auth_service.create_api_key(user.id, data)

    return APIKeyCreated(
        id=api_key.id,
        name=api_key.name,
        key=plain_key,  # Only time the full key is returned!
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        rate_limit_tier=api_key.rate_limit_tier,
        scopes=api_key.scope_list,
        expires_at=api_key.expires_at,
        last_used_at=api_key.last_used_at,
        created_at=api_key.created_at,
    )


@router.get(
    "/api-keys",
    response_model=list[APIKeyResponse],
    summary="List API keys",
    description="List all API keys for the current user.",
)
async def list_api_keys(
    user: CurrentUserDep,
    auth_service: AuthServiceDep,
) -> list[APIKeyResponse]:
    """List all API keys for the authenticated user."""
    api_keys = await auth_service.list_user_api_keys(user.id)

    return [
        APIKeyResponse(
            id=key.id,
            name=key.name,
            key_prefix=key.key_prefix,
            is_active=key.is_active,
            rate_limit_tier=key.rate_limit_tier,
            scopes=key.scope_list,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            created_at=key.created_at,
        )
        for key in api_keys
    ]


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Deactivate an API key so it can no longer be used.",
)
async def revoke_api_key(
    key_id: int,
    user: CurrentUserDep,
    auth_service: AuthServiceDep,
) -> None:
    """
    Revoke an API key.

    The key is deactivated, not deleted, for audit purposes.
    """
    success = await auth_service.revoke_api_key(user.id, key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )