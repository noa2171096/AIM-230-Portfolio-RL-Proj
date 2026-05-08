"""
Redis Caching Service

This module provides a caching layer using Redis for:
- Expensive query results
- ML model outputs
- User session data
- Rate limiting counters

Features:
- Async Redis operations
- JSON serialization
- TTL-based expiration
- Cache invalidation helpers
"""

import json
import logging
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, TypeVar

import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheService:
    """
    Redis-based caching service.

    Provides async get/set operations with JSON serialization.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._redis: Redis | None = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis.url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Connected to Redis cache")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis cache")

    @property
    def redis(self) -> Redis:
        """Get Redis client, raising if not connected."""
        if self._redis is None:
            raise RuntimeError("Cache not connected. Call connect() first.")
        return self._redis

    async def get(self, key: str) -> Any | None:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Deserialized value or None if not found
        """
        try:
            value = await self.redis.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | timedelta | None = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds or timedelta

        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value)
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())

            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "user:123:*")

        Returns:
            Number of keys deleted
        """
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern error for {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for {key}: {e}")
            return False

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Cache incr error for {key}: {e}")
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on an existing key."""
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.warning(f"Cache expire error for {key}: {e}")
            return False


# Cache key builders
class CacheKeys:
    """Standardized cache key builders."""

    @staticmethod
    def user(user_id: int) -> str:
        return f"user:{user_id}"

    @staticmethod
    def user_assets(user_id: int) -> str:
        return f"user:{user_id}:assets"

    @staticmethod
    def user_embeddings(user_id: int) -> str:
        return f"user:{user_id}:embeddings"

    @staticmethod
    def asset(asset_id: int) -> str:
        return f"asset:{asset_id}"

    @staticmethod
    def asset_detail(asset_id: int) -> str:
        return f"asset:{asset_id}:detail"

    @staticmethod
    def search_results(user_id: int, query_hash: str) -> str:
        return f"search:{user_id}:{query_hash}"

    @staticmethod
    def rate_limit(identifier: str) -> str:
        return f"ratelimit:{identifier}"


# Default TTLs
class CacheTTL:
    """Standard TTL values."""

    SHORT = 60  # 1 minute
    MEDIUM = 300  # 5 minutes
    LONG = 3600  # 1 hour
    DAY = 86400  # 24 hours

    # Specific TTLs
    USER_DATA = MEDIUM
    ASSET_LIST = SHORT
    ASSET_DETAIL = MEDIUM
    SEARCH_RESULTS = SHORT
    EMBEDDINGS = LONG


# Global cache instance
_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        raise RuntimeError("Cache service not initialized. Call init_cache() first.")
    return _cache_service


async def init_cache(settings: Settings | None = None) -> CacheService:
    """Initialize the cache service (call during startup)."""
    global _cache_service
    _cache_service = CacheService(settings)
    await _cache_service.connect()
    return _cache_service


async def close_cache() -> None:
    """Close the cache connection (call during shutdown)."""
    global _cache_service
    if _cache_service:
        await _cache_service.disconnect()
        _cache_service = None


def cached(
    key_builder: Callable[..., str],
    ttl: int = CacheTTL.MEDIUM,
    skip_cache: Callable[..., bool] | None = None,
):
    """
    Decorator for caching function results.

    Args:
        key_builder: Function to build cache key from arguments
        ttl: Time-to-live in seconds
        skip_cache: Optional function to determine if cache should be skipped

    Example:
        @cached(
            key_builder=lambda user_id: f"user:{user_id}:profile",
            ttl=300,
        )
        async def get_user_profile(user_id: int):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Check if we should skip cache
            if skip_cache and skip_cache(*args, **kwargs):
                return await func(*args, **kwargs)

            cache = get_cache_service()
            key = key_builder(*args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {key}")
                return cached_value

            # Call function and cache result
            logger.debug(f"Cache miss: {key}")
            result = await func(*args, **kwargs)

            if result is not None:
                await cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


def invalidate_on_change(key_patterns: list[Callable[..., str]]):
    """
    Decorator to invalidate cache when data changes.

    Args:
        key_patterns: List of functions that generate keys to invalidate

    Example:
        @invalidate_on_change([
            lambda user_id: f"user:{user_id}:*",
        ])
        async def update_user(user_id: int, data: dict):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            result = await func(*args, **kwargs)

            # Invalidate cache
            cache = get_cache_service()
            for pattern_builder in key_patterns:
                pattern = pattern_builder(*args, **kwargs)
                await cache.delete_pattern(pattern)

            return result

        return wrapper
    return decorator