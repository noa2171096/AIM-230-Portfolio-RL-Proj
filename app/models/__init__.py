# /app/models/__init__.py

"""
SQLAlchemy Models Package

All models are imported here to ensure they are registered
with SQLAlchemy's metadata before Alembic runs migrations.
"""

from app.models.base import Base, TimestampMixin
from app.models.user import APIKey, User
from app.models.asset import Asset, AssetStatus
#from app.models.tag import UserTag

__all__ = [
    "Base",
    "TimestampMixin",
    "User",
    "APIKey",
    "Asset",
    "AssetStatus"
]