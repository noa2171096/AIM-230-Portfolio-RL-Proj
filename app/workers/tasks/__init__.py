# app/workers/tasks/__init__.py
from app.workers.tasks.processing import generate_portfolio_recommendation

__all__ = ["generate_portfolio_recommendation"]