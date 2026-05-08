# app/api/v1/assets.py
"""
Asset Upload and Management Endpoints

Supports upload and retrieval of images and PDFs.
No ML processing yet — storage and tracking only.
Future: AI evaluation of financial documents.
"""

import io
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.api.v1.auth import CurrentUserDep, CurrentUserOptionalDep
from app.config import get_settings
from app.database import DbSessionDep
from app.models.asset import Asset, AssetStatus
from app.schemas.asset import (
    AssetDetail,
    AssetList,
    AssetResponse,
    AssetUploadResponse,
)
from app.services.storage import get_storage_service

router   = APIRouter()
settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────
def build_asset_url(asset: Asset) -> str:
    return f"{settings.api_v1_prefix}/assets/{asset.id}/file"


def asset_to_response(asset: Asset) -> AssetResponse:
    return AssetResponse(
        id                = asset.id,
        filename          = asset.filename,
        original_filename = asset.original_filename,
        content_type      = asset.content_type,
        file_size         = asset.file_size,
        status            = asset.status,
        created_at        = asset.created_at,
        url               = build_asset_url(asset),
    )


def asset_to_detail(asset: Asset) -> AssetDetail:
    return AssetDetail(
        id                = asset.id,
        filename          = asset.filename,
        original_filename = asset.original_filename,
        content_type      = asset.content_type,
        file_size         = asset.file_size,
        status            = asset.status,
        created_at        = asset.created_at,
        processed_at      = asset.processed_at,
        url               = build_asset_url(asset),
        error_message     = asset.error_message,
    )


# ── Allowed types ─────────────────────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "application/pdf",
}


# ── Upload ────────────────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model = AssetUploadResponse,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Upload a file",
    description    = "Upload an image or PDF. No processing — storage only.",
)
async def upload_asset(
    user: CurrentUserDep,
    db:   DbSessionDep,
    file: UploadFile = File(...),
) -> AssetUploadResponse:
    """
    Upload a single image or PDF.
    File is stored and tracked in the database.
    No ML processing is triggered.
    """
    # Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"File type '{file.content_type}' not allowed. "
                          f"Allowed: {list(ALLOWED_TYPES)}",
        )

    # Read + validate size
    contents = await file.read()
    size_mb  = len(contents) / (1024 * 1024)

    if size_mb > settings.storage.max_file_size_mb:
        raise HTTPException(
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail      = f"File too large: {size_mb:.1f}MB. "
                          f"Max: {settings.storage.max_file_size_mb}MB",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Empty file",
        )

    # Save to storage
    storage      = get_storage_service()
    file_obj     = io.BytesIO(contents)
    storage_path = await storage.save_file(
        file         = file_obj,
        filename     = file.filename or "unnamed",
        content_type = file.content_type,
        user_id      = user.id,
    )

    # Create database record
    asset = Asset(
        user_id           = user.id,
        filename          = storage_path.split("/")[-1],
        original_filename = file.filename or "unnamed",
        content_type      = file.content_type,
        file_size         = len(contents),
        storage_path      = storage_path,
        status            = AssetStatus.COMPLETED.value,  # no processing needed
    )

    db.add(asset)
    await db.flush()

    return AssetUploadResponse(
        id                = asset.id,
        filename          = asset.filename,
        original_filename = asset.original_filename,
        content_type      = asset.content_type,
        file_size         = asset.file_size,
        status            = asset.status,
        message           = "File uploaded successfully.",
    )


# ── Batch upload ──────────────────────────────────────────────────────────
@router.post(
    "/upload/batch",
    response_model = List[AssetUploadResponse],
    status_code    = status.HTTP_201_CREATED,
    summary        = "Upload multiple files",
)
async def upload_assets_batch(
    user:  CurrentUserDep,
    db:    DbSessionDep,
    files: List[UploadFile] = File(...),
) -> List[AssetUploadResponse]:
    if len(files) > 10:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Maximum 10 files per batch",
        )

    results = []
    errors  = []

    for i, file in enumerate(files):
        try:
            result = await upload_asset(user, db, file)
            results.append(result)
        except HTTPException as e:
            errors.append({
                "index":    i,
                "filename": file.filename,
                "error":    e.detail,
            })

    if errors and not results:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = {"message": "All uploads failed", "errors": errors},
        )

    return results


# ── List ──────────────────────────────────────────────────────────────────
@router.get(
    "",
    response_model = AssetList,
    summary        = "List uploaded files",
)
async def list_assets(
    user:      CurrentUserDep,
    db:        DbSessionDep,
    page:      int = Query(1,  ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> AssetList:
    from sqlalchemy import func, select

    query = select(Asset).where(Asset.user_id == user.id)

    count_query = select(func.count()).select_from(query.subquery())
    total       = (await db.execute(count_query)).scalar() or 0

    offset  = (page - 1) * page_size
    query   = query.order_by(Asset.created_at.desc()).offset(offset).limit(page_size)
    result  = await db.execute(query)
    assets  = result.scalars().all()
    pages   = (total + page_size - 1) // page_size if total > 0 else 1

    return AssetList(
        items     = [asset_to_response(a) for a in assets],
        total     = total,
        page      = page,
        page_size = page_size,
        pages     = pages,
    )


# ── Detail ────────────────────────────────────────────────────────────────
@router.get(
    "/{asset_id}",
    response_model = AssetDetail,
    summary        = "Get file details",
)
async def get_asset(
    asset_id: int,
    user:     CurrentUserDep,
    db:       DbSessionDep,
) -> AssetDetail:
    asset = await db.get(Asset, asset_id)
    if not asset or asset.user_id != user.id:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset_to_detail(asset)


# ── View / Download ───────────────────────────────────────────────────────
@router.get(
    "/{asset_id}/file",
    summary = "View or download file",
)
async def download_asset(
    asset_id: int,
    db:       DbSessionDep,
    user:     CurrentUserOptionalDep = None,
    token:    Optional[str] = Query(None, description="JWT token for browser requests"),
) -> FileResponse:
    # Handle token auth for browser image loading
    if not user and token:
        from app.services.auth import AuthService
        auth_service = AuthService(db)
        user_id      = auth_service.verify_access_token(token)
        if user_id:
            user = await auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    asset = await db.get(Asset, asset_id)
    if not asset or asset.user_id != user.id:
        raise HTTPException(status_code=404, detail="Asset not found")

    storage   = get_storage_service()
    file_path = await storage.get_file_path(asset.storage_path)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found in storage")

    return FileResponse(
        path       = file_path,
        media_type = asset.content_type,
        filename   = asset.original_filename,
    )


# ── Delete ────────────────────────────────────────────────────────────────
@router.delete(
    "/{asset_id}",
    status_code = status.HTTP_204_NO_CONTENT,
    summary     = "Delete a file",
)
async def delete_asset(
    asset_id: int,
    user:     CurrentUserDep,
    db:       DbSessionDep,
) -> None:
    asset = await db.get(Asset, asset_id)
    if not asset or asset.user_id != user.id:
        raise HTTPException(status_code=404, detail="Asset not found")

    storage = get_storage_service()
    await storage.delete_file(asset.storage_path)
    await db.delete(asset)