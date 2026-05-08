# app/workers/tasks/processing.py
import os
import pickle
from celery import shared_task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app
from app.config import get_settings

logger   = get_task_logger(__name__)
settings = get_settings()


@celery_app.task(
    bind                = True,
    name                = "tasks.generate_portfolio_recommendation",
    max_retries         = 2,
    default_retry_delay = 30,
    track_started       = True,
)
def generate_portfolio_recommendation(
    self,
    tickers:       list,
    horizon_days:  int   = 21,
    episodes:      int   = 200,
    capital:       float = 10_000.0,
    force_retrain: bool  = False,
):
    """
    Celery background task that runs the full RL portfolio pipeline.

    """
    try:
        logger.info(
            f"Starting portfolio task | "
            f"tickers={tickers} | "
            f"horizon={horizon_days}d | "
            f"episodes={episodes}"
        )

        # ── Stage 1: Load data ────────────────────────────────────────────
        self.update_state(
            state = "PROGRESS",
            meta  = {
                "step":    "loading_data",
                "tickers": tickers,
                "message": "Loading market datasets",
            }
        )

        preloaded = None
        dataset_path = settings.ml.dataset_path

        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                preloaded = pickle.load(f)
            logger.info(f"Loaded cached datasets: {list(preloaded.keys())}")
        else:
            logger.warning(
                f"No cached dataset at {dataset_path} — "
                f"will download via build_features"
            )

        # ── Stage 2: Train / load agent ───────────────────────────────────
        self.update_state(
            state = "PROGRESS",
            meta  = {
                "step":    "training",
                "tickers": tickers,
                "message": f"Training agent ({episodes} episodes)",
            }
        )

        # Import here — avoids loading heavy ML libs at worker startup
        from app.ml.frontend import generate_recommendation

        result = generate_recommendation(
            user_tickers       = tickers,
            horizon_days       = horizon_days,
            episodes           = episodes,
            capital            = capital,
            force_retrain      = force_retrain,
            preloaded_datasets = preloaded,
        )

        logger.info(
            f"Task complete | "
            f"return={result['portfolio']['expected_return_pct']}% | "
            f"sharpe={result['portfolio']['sharpe_ratio']}"
        )

        return result

    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise self.retry(exc=e)