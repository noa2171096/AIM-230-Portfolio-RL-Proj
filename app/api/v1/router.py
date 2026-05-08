# router.py
from app.api.v1 import portfolio
from fastapi import APIRouter
from app.api.v1 import assets, auth, health, portfolio, files

api_router = APIRouter()

api_router.include_router(health.router,     prefix="/health",    tags=["Health"])
api_router.include_router(auth.router,       prefix="/auth",      tags=["Authentication"])
api_router.include_router(assets.router,     prefix="/assets",    tags=["Assets"])
#api_router.include_router(search.router,     prefix="/search",    tags=["Search"])
api_router.include_router(portfolio.router,   prefix="/analysis",  tags=["Analysis & MLOps"])
#api_router.include_router(tags.router,       prefix="/tags",      tags=["Tags"])
api_router.include_router(portfolio.router,  prefix="/portfolio", tags=["Portfolio RL"])  # ← new
#api_router.include_router(files.router, prefix="/files", tags=["Files"])
#files.py is simple implementation of assets.py