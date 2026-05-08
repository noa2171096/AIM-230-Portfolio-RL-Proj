"""
Adapted from Visual Vault 

Celery Application Configuration

This module demonstrates:
- Celery app initialization
- Task routing and queues
- Retry policies
- Result backend configuration
"""

from celery import Celery

from app.config import get_settings

settings = get_settings()


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application.

    This factory pattern allows for different configurations
    in testing vs production.
    """
    print(f"Redis URL: {settings.redis.url}")
    print(f"Redis host: {settings.redis.host}")
    
    app = Celery(
        "Portfolio RL",
        broker=settings.redis.url,
        backend=settings.redis.url,
        include=[
            "app.workers.tasks.processing",
        ],
    )

    # Celery configuration
    app.conf.update(
        # Task settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,

        # Task execution
        task_always_eager=settings.celery.task_always_eager,
        task_eager_propagates=settings.celery.task_eager_propagates,
        task_time_limit=settings.celery.task_time_limit,
        task_soft_time_limit=settings.celery.task_soft_time_limit,

        # Worker settings
        worker_concurrency=settings.celery.worker_concurrency,
        worker_prefetch_multiplier=1,  # Fetch one task at a time

        # Result settings
        result_expires=3600,  # Results expire after 1 hour

        # Task routing
        task_routes={
            "app.workers.tasks.processing.*": {"queue": "ml"},
        },

        # Default queue
        task_default_queue="default",

        # Retry policy
        task_acks_late=True,  # Acknowledge after task completes
        task_reject_on_worker_lost=True,  # Requeue if worker dies
    )

    return app


# Create the Celery app instance
celery_app = create_celery_app()


# Optional: Configure periodic tasks (Celery Beat)
# celery_app.conf.beat_schedule = {
#     "cleanup-old-assets": {
#         "task": "app.workers.tasks.maintenance.cleanup_old_assets",
#         "schedule": 3600.0,  # Every hour
#     },
# }