# app/api/v1/portfolio.py
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
import pickle
import os
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../ml"))

from app.ml.frontend import generate_recommendation

router = APIRouter()

# ── In-memory job store (swap for Redis in production) ────────────────────
jobs: Dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    tickers:       List[str]
    horizon_days:  Optional[int]   = 21
    episodes:      Optional[int]   = 200
    capital:       Optional[float] = 10_000.0
    force_retrain: Optional[bool]  = False

    @validator("tickers")
    def validate_tickers(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 tickers")
        if len(v) > 10:
            raise ValueError("Maximum 10 tickers")
        return [t.upper().strip() for t in v]

    @validator("horizon_days")
    def validate_horizon(cls, v):
        if v not in [5, 21, 63]:
            raise ValueError("horizon_days must be 5 (1W), 21 (1M), or 63 (3M)")
        return v


class AllocationItem(BaseModel):
    weight:        float
    weight_pct:    float
    dollar_value:  float
    action:        str
    current_price: float
    shares_to_buy: float


class PortfolioMetrics(BaseModel):
    expected_return_pct: float
    sharpe_ratio:        float
    max_drawdown_pct:    float
    final_value:         float


class RecommendationResponse(BaseModel):
    job_id:       str
    status:       str
    tickers:      Optional[List[str]]              = None
    allocation:   Optional[Dict[str, AllocationItem]] = None
    portfolio:    Optional[PortfolioMetrics]        = None
    horizon_days: Optional[int]                    = None
    generated_at: Optional[str]                    = None
    error:        Optional[str]                    = None


# ── Routes ────────────────────────────────────────────────────────────────
@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    req:              RecommendationRequest,
    background_tasks: BackgroundTasks,
    request:          Request,             
):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status":     "running",
        "started_at": datetime.now().isoformat(),
    }

    # Use preloaded datasets from app state — no disk read per request
    preloaded = request.app.state.datasets  # ← use app.state

    background_tasks.add_task(
        run_recommendation_job,
        job_id    = job_id,
        req       = req,
        preloaded = preloaded,
    )

    return RecommendationResponse(job_id=job_id, status="running")


@router.get("/recommend/{job_id}", response_model=RecommendationResponse)
def get_recommendation(job_id: str):
    """Poll this endpoint to check if your recommendation is ready."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]

    if job["status"] == "running":
        return RecommendationResponse(job_id=job_id, status="running")

    if job["status"] == "failed":
        return RecommendationResponse(
            job_id=job_id,
            status="failed",
            error=job.get("error"),
        )

    result = job["result"]
    return RecommendationResponse(
        job_id       = job_id,
        status       = "complete",
        tickers      = result["tickers"],
        allocation   = result["allocation"],
        portfolio    = result["portfolio"],
        horizon_days = result["horizon_days"],
        generated_at = result["generated_at"],
    )


@router.get("/jobs")
def list_jobs():
    """List all submitted jobs and their statuses."""
    return {
        jid: {
            "status":     j["status"],
            "started_at": j.get("started_at"),
        }
        for jid, j in jobs.items()
    }


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs[job_id]
    return {"deleted": job_id}


# ── Background task ───────────────────────────────────────────────────────
def run_recommendation_job(job_id, req, preloaded):
    try:
        result = generate_recommendation(
            user_tickers       = req.tickers,
            horizon_days       = req.horizon_days,
            episodes           = req.episodes,
            capital            = req.capital,
            force_retrain      = req.force_retrain,
            preloaded_datasets = preloaded,
        )
        jobs[job_id] = {"status": "complete", "result": result}

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}