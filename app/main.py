"""
RL Portfolio Advisor - Main Application Entry Point
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.metrics import MetricsMiddleware, get_metrics_collector
from app.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from app.api.v1.router import api_router
from app.config import Settings, get_settings
from app.services.cache import init_cache, close_cache
from app.database import init_db, close_db

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
        if not get_settings().debug
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown lifecycle manager.

    Startup:
    - Creates required directories
    - Pre-loads FinBERT model so first request isn't slow
    - Verifies dataset exists

    Shutdown:
    - Cleans up any open resources
    """
    settings = get_settings()

    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info(
        "Starting RL Portfolio Advisor",
        environment = settings.environment,
        debug       = settings.debug,
    )

    # Ensure required directories exist
    import os
    for directory in ["data", "saved_models", "app/ml/models/finbert"]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories initialized")

        # Initialize database connection pool
    init_db(settings)
    logger.info("Database connection pool initialized")

    await init_cache(settings)
    logger.info("Cache service initialized")

    # Pre-load FinBERT so first recommendation request isn't slow
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        finbert_path = "app/ml/models/finbert"

        if os.path.exists(os.path.join(finbert_path, "model.safetensors")) or os.path.exists(os.path.join(finbert_path, "pytorch_model.bin")):
            logger.info("Loading FinBERT from local cache...")
            tokenizer = BertTokenizer.from_pretrained(finbert_path)
            model     = BertForSequenceClassification.from_pretrained(finbert_path)
        else:
            logger.info("Downloading FinBERT (first run — ~438MB)...")
            tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
            tokenizer.save_pretrained(finbert_path)
            model.save_pretrained(finbert_path)

        # Store on app state so routes can access without reloading
        app.state.finbert_tokenizer = tokenizer
        app.state.finbert_model     = model
        logger.info("FinBERT loaded")

    except Exception as e:
        logger.warning(f"FinBERT failed to load: {e} — sentiment features will be zero")
        import traceback
        traceback.print_exc()   
        app.state.finbert_tokenizer = None
        app.state.finbert_model     = None

    # Verify dataset exists
    import pickle
    dataset_path = "data/datasets_2015-01-01_2023-12-31.pkl"
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            datasets = pickle.load(f)
        app.state.datasets = datasets
        logger.info(
            "Datasets loaded",
            tickers = list(datasets.keys()),
            rows    = {t: len(df) for t, df in datasets.items()},
        )
    else:
        app.state.datasets = None
        logger.warning(
            "No dataset found at data/datasets_2015-01-01_2023-12-31.pkl "
            "— will download on first request"
        )

    yield  # Application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("Shutting down RL Portfolio Advisor")

    await close_cache()
    logger.info("Cache connections closed")

    # Close database connections
    await close_db()
    logger.info("Database connections closed")
    
    app.state.datasets          = None
    app.state.finbert_tokenizer = None
    app.state.finbert_model     = None
    logger.info("Resources released")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Application factory.
    Allows easy testing with different configs.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title       = settings.app_name,
        version     = settings.app_version,
        description = """
## RL Portfolio Advisor

Deep Q-Learning agent for portfolio optimization and near-term allocation recommendations.

### Features
- **Portfolio Recommendations**: RL-optimised allocation across any set of tickers
- **Multiple Horizons**: 1 week, 1 month, 3 month outlooks
- **Async Training**: Background job system — submit and poll for results
- **Cached Models**: Trained agents saved per ticker set — no retraining needed
- **Sentiment Features**: FinBERT-scored news integrated into state space
        """,
        docs_url    = "/docs"         if settings.debug else None,
        redoc_url   = "/redoc"        if settings.debug else None,
        openapi_url = "/openapi.json" if settings.debug else None,
        lifespan    = lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )
    # Add custom middleware (order matters - first added = outermost)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # Configure rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


# Create application instance
app = create_app()

@app.get("/")
def root():
    return {
        "name":    "RL Portfolio Advisor",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }