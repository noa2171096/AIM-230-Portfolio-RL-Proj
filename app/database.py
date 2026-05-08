"""
Adapted from Visual Vault

Database Configuration and Session Management

This module demonstrates:
- Async SQLAlchemy engine and session setup
- Session dependency for FastAPI
- Connection pool configuration
- Database health checking
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings


def create_engine(settings: Settings):
    """
    Create an async SQLAlchemy engine.

    The engine manages the connection pool and is typically created once
    at application startup.

    Args:
        settings: Application settings containing database configuration

    Returns:
        AsyncEngine: Configured async database engine
    """
    return create_async_engine(
        settings.database.url,
        echo=settings.database.echo,  # Log SQL statements if True
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        # For testing, you might want NullPool to avoid connection issues
        # poolclass=NullPool,
    )


def create_session_maker(engine) -> async_sessionmaker[AsyncSession]:
    """
    Create a session factory.

    The session maker is used to create individual database sessions.
    Each request typically gets its own session.

    Args:
        engine: The SQLAlchemy async engine

    Returns:
        async_sessionmaker: Factory for creating async sessions
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autocommit=False,
        autoflush=False,
    )


# Global engine and session maker (initialized at startup)
# These will be set by the lifespan context manager
_engine = None
_async_session_maker = None


def init_db(settings: Settings) -> None:
    """
    Initialize the database engine and session maker.

    Called during application startup.
    """
    global _engine, _async_session_maker
    _engine = create_engine(settings)
    _async_session_maker = create_session_maker(_engine)


async def close_db() -> None:
    """
    Close the database engine and all connections.

    Called during application shutdown.
    """
    global _engine
    if _engine:
        await _engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Usage:
        @app.get("/items")
        async def get_items(db: DbSessionDep):
            result = await db.execute(select(Item))
            return result.scalars().all()

    The session is automatically closed after the request completes.
    If an exception occurs, the session is rolled back.

    Yields:
        AsyncSession: Database session for the current request
    """
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Type alias for dependency injection
DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


async def check_db_connection() -> bool:
    """
    Check if the database is reachable.

    Used by health check endpoints.

    Returns:
        bool: True if database is connected, False otherwise
    """
    from sqlalchemy import text

    if _async_session_maker is None:
        return False

    try:
        async with _async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception:
        return False