"""
Application Configuration using Pydantic Settings

This module demonstrates:
- Environment-based configuration with Pydantic Settings
- Type validation for config values
- Nested configuration models
- Computed properties for derived values

Pydantic Settings automatically reads from:
1. Environment variables
2. .env file (if python-dotenv is installed)
3. Default values defined in the model
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    user: str = "RLfin"
    password: str = "RLfin"
    name: str = "RLfin"

    echo: bool = False  # SQLAlchemy echo SQL statements
    pool_size: int = 5
    max_overflow: int = 10

    @computed_field
    @property
    def url(self) -> str:
        """Construct the database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @computed_field
    @property
    def url_sync(self) -> str:
        """Sync URL for Alembic migrations."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and Celery broker."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @computed_field
    @property
    def url(self) -> str:
        """Construct the Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class StorageSettings(BaseSettings):
    """File storage configuration."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # Base path for all file storage
    base_path: Path = Path("storage")

    # Subdirectories
    uploads_dir: str = "uploads"
    embeddings_dir: str = "embeddings"

    # Upload constraints
    max_file_size_mb: int = 10
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".pdf"}

    @computed_field
    @property
    def uploads_path(self) -> Path:
        """Full path to uploads directory."""
        return self.base_path / self.uploads_dir

    @computed_field
    @property
    def embeddings_path(self) -> Path:
        """Full path to embeddings directory."""
        return self.base_path / self.embeddings_dir

    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        """Max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


class AuthSettings(BaseSettings):
    """Authentication and security configuration."""

    model_config = SettingsConfigDict(env_prefix="AUTH_")

    # JWT Settings
    secret_key: str = "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # API Key Settings
    api_key_prefix: str = "RLfin_"  # RL portfolio prefix
    api_key_length: int = 32

    # Rate Limiting (requests per minute)
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10


class MLSettings(BaseSettings):
    """Machine Learning model configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # Model paths
    models_dir: Path = Path("models")

    # CLIP settings
    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_dimension: int = 512

    # Device configuration
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Inference settings
    batch_size: int = 8
    num_workers: int = 2


class CelerySettings(BaseSettings):
    """Celery task queue configuration."""

    model_config = SettingsConfigDict(env_prefix="CELERY_")

    task_always_eager: bool = False  # Set True for testing (runs tasks synchronously)
    task_eager_propagates: bool = True
    worker_concurrency: int = 2
    task_time_limit: int = 300  # 5 minutes max per task
    task_soft_time_limit: int = 240  # Soft limit for graceful shutdown


class Settings(BaseSettings):
    """
    Main application settings.

    All settings can be overridden via environment variables.
    Nested settings use their own prefixes (e.g., DB_HOST, REDIS_PORT).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

    # Application metadata
    app_name: str = "RL Portfolio Advisor"
    app_version: str = "1.0.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"

    # API Settings
    api_v1_prefix: str = "/api/v1"

    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Nested configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Create and cache settings instance.

    Using lru_cache ensures we only create one Settings instance,
    which is important because reading from .env file and environment
    variables should only happen once at startup.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()