# app/api/v1/files.py
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

# ── Storage directory ─────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Allowed file types ────────────────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg":       ".jpg",
    "image/png":        ".png",
    "image/gif":        ".gif",
    "image/webp":       ".webp",
    "application/pdf":  ".pdf",
}

MAX_FILE_SIZE_MB = 10


# ── Response models ───────────────────────────────────────────────────────
class FileInfo(BaseModel):
    file_id:   str
    filename:  str
    file_type: str
    size_kb:   float
    url:       str


class FileListResponse(BaseModel):
    files: List[FileInfo]
    total: int


# ── Routes ────────────────────────────────────────────────────────────────
@router.post("/upload", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an image or PDF file.
    Returns a file_id you can use to view or delete it later.
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed: {file.content_type}. "
                   f"Allowed: {list(ALLOWED_TYPES.keys())}"
        )

    # Read file
    contents = await file.read()

    # Validate file size
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB"
        )

    # Generate unique ID and save
    file_id  = str(uuid.uuid4())[:8]
    ext      = ALLOWED_TYPES[file.content_type]
    filename = f"{file_id}{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        f.write(contents)

    return FileInfo(
        file_id   = file_id,
        filename  = file.filename or filename,
        file_type = file.content_type,
        size_kb   = round(len(contents) / 1024, 2),
        url       = f"/api/v1/files/{file_id}",
    )


@router.get("/", response_model=FileListResponse)
def list_files():
    """List all uploaded files."""
    files = []
    for filepath in UPLOAD_DIR.iterdir():
        if filepath.is_file():
            file_id  = filepath.stem
            ext      = filepath.suffix
            size_kb  = round(filepath.stat().st_size / 1024, 2)

            # Determine content type from extension
            type_map = {v: k for k, v in ALLOWED_TYPES.items()}
            content_type = type_map.get(ext, "application/octet-stream")

            files.append(FileInfo(
                file_id   = file_id,
                filename  = filepath.name,
                file_type = content_type,
                size_kb   = size_kb,
                url       = f"/api/v1/files/{file_id}",
            ))

    return FileListResponse(files=files, total=len(files))


@router.get("/{file_id}")
def view_file(file_id: str):
    """
    View or download a file by its file_id.
    Images render in browser, PDFs open in PDF viewer.
    """
    # Find the file with any extension
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    filepath = matches[0]
    ext      = filepath.suffix

    type_map     = {v: k for k, v in ALLOWED_TYPES.items()}
    content_type = type_map.get(ext, "application/octet-stream")

    return FileResponse(
        path         = filepath,
        media_type   = content_type,
        filename     = filepath.name,
    )


@router.delete("/{file_id}")
def delete_file(file_id: str):
    """Delete a file by its file_id."""
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    matches[0].unlink()
    return {"deleted": file_id}