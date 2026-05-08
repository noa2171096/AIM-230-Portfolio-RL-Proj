"""
Celery workers package.

This package contains the Celery application configuration
and background task definitions.
"""

from app.workers.celery_app import celery_app

__all__ = ["celery_app"]