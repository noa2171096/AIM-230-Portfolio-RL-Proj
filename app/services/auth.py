"""
Authentication Service

This module demonstrates:
- Password hashing with bcrypt
- JWT token creation and verification
- API key generation and validation
- User authentication logic
"""

import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AuthSettings, get_settings
from app.models.user import APIKey, User
from app.schemas.user import APIKeyCreate, UserCreate

# Password hashing context using bcrypt
# bcrypt automatically handles salting
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """
    Service for authentication operations.

    Handles:
    - User registration and login
    - Password hashing and verification
    - JWT token management
    - API key management
    """

    def __init__(self, db: AsyncSession, auth_settings: AuthSettings | None = None):
        """
        Initialize auth service.

        Args:
            db: Database session
            auth_settings: Optional auth settings (uses default if not provided)
        """
        self.db = db
        self.settings = auth_settings or get_settings().auth

    # =========================================================================
    # Password Operations
    # =========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored password hash

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    # =========================================================================
    # User Operations
    # =========================================================================

    async def get_user_by_email(self, email: str) -> User | None:
        """
        Find a user by email address.

        Args:
            email: Email to search for

        Returns:
            User if found, None otherwise
        """
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int) -> User | None:
        """
        Find a user by ID.

        Args:
            user_id: User ID to search for

        Returns:
            User if found, None otherwise
        """
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_user(self, data: UserCreate) -> User:
        """
        Create a new user account.

        Args:
            data: User registration data

        Returns:
            Created user

        Raises:
            ValueError: If email already exists
        """
        # Check if email already exists
        existing = await self.get_user_by_email(data.email)
        if existing:
            raise ValueError("Email already registered")

        # Create user with hashed password
        user = User(
            email=data.email.lower(),
            hashed_password=self.hash_password(data.password),
            full_name=data.full_name,
            is_active=True,
            is_verified=False,  # Would need email verification in production
        )

        self.db.add(user)
        await self.db.flush()  # Get the ID without committing
        await self.db.refresh(user)

        return user

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """
        Authenticate a user with email and password.

        Args:
            email: User's email
            password: Plain text password

        Returns:
            User if credentials are valid, None otherwise
        """
        user = await self.get_user_by_email(email)

        if not user:
            # Still hash to prevent timing attacks
            self.hash_password(password)
            return None

        if not self.verify_password(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        return user

    # =========================================================================
    # JWT Token Operations
    # =========================================================================

    def create_access_token(
        self,
        user_id: int,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User ID to encode in token
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.settings.access_token_expire_minutes)

        expire = datetime.now(timezone.utc) + expires_delta

        payload = {
            "sub": str(user_id),  # Subject (user ID)
            "exp": expire,  # Expiration time
            "iat": datetime.now(timezone.utc),  # Issued at
            "type": "access",  # Token type
        }

        return jwt.encode(
            payload,
            self.settings.secret_key,
            algorithm=self.settings.algorithm,
        )

    def verify_access_token(self, token: str) -> int | None:
        """
        Verify a JWT access token and extract user ID.

        Args:
            token: JWT token string

        Returns:
            User ID if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
            )

            user_id = payload.get("sub")
            token_type = payload.get("type")

            if user_id is None or token_type != "access":
                return None

            return int(user_id)

        except JWTError:
            return None

    # =========================================================================
    # API Key Operations
    # =========================================================================

    def generate_api_key(self) -> str:
        """
        Generate a new API key.

        Format: {prefix}_{random_string}
        Example: vv_abc123def456ghi789jkl012mno345pq

        Returns:
            New API key string
        """
        prefix = self.settings.api_key_prefix
        random_part = secrets.token_urlsafe(self.settings.api_key_length)
        return f"{prefix}{random_part}"

    async def create_api_key(self, user_id: int, data: APIKeyCreate) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.

        Args:
            user_id: ID of the user creating the key
            data: API key creation data

        Returns:
            Tuple of (APIKey model, plain text key)

        Note:
            The plain text key is only returned once!
            We store only the hash.
        """
        # Generate the key
        plain_key = self.generate_api_key()
        key_prefix = plain_key[: len(self.settings.api_key_prefix) + 8]

        # Calculate expiration if specified
        expires_at = None
        if data.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_in_days)

        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            name=data.name,
            key_prefix=key_prefix,
            key_hash=self.hash_password(plain_key),  # Reuse password hashing
            is_active=True,
            scopes=",".join(data.scopes),
            expires_at=expires_at,
        )

        self.db.add(api_key)
        await self.db.flush()
        await self.db.refresh(api_key)

        return api_key, plain_key

    async def verify_api_key(self, key: str) -> tuple[APIKey, User] | None:
        """
        Verify an API key and return the associated key and user.

        Args:
            key: Plain text API key

        Returns:
            Tuple of (APIKey, User) if valid, None otherwise
        """
        # Extract prefix for lookup
        prefix = key[: len(self.settings.api_key_prefix) + 8]

        # Find potential matches by prefix
        result = await self.db.execute(
            select(APIKey)
            .where(APIKey.key_prefix == prefix)
            .where(APIKey.is_active == True)  # noqa: E712
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return None

        # Verify the full key
        if not self.verify_password(key, api_key.key_hash):
            return None

        # Check expiration
        if api_key.is_expired:
            return None

        # Update last used timestamp
        api_key.last_used_at = datetime.now(timezone.utc)

        # Load the user
        user_result = await self.db.execute(
            select(User).where(User.id == api_key.user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user or not user.is_active:
            return None

        return api_key, user

    async def revoke_api_key(self, user_id: int, key_id: int) -> bool:
        """
        Revoke (deactivate) an API key.

        Args:
            user_id: ID of the user who owns the key
            key_id: ID of the key to revoke

        Returns:
            True if key was revoked, False if not found
        """
        result = await self.db.execute(
            select(APIKey)
            .where(APIKey.id == key_id)
            .where(APIKey.user_id == user_id)
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return False

        api_key.is_active = False
        return True

    async def list_user_api_keys(self, user_id: int) -> list[APIKey]:
        """
        List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API keys
        """
        result = await self.db.execute(
            select(APIKey)
            .where(APIKey.user_id == user_id)
            .order_by(APIKey.created_at.desc())
        )
        return list(result.scalars().all())