"""
Storage Service Abstraction

Provides a unified interface for file storage operations.
Supports local filesystem storage with an architecture that
can be extended for cloud storage (S3, GCS, Azure Blob).
"""

import hashlib
import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from app.config import Settings, get_settings


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
        user_id: int,
    ) -> str:
        """
        Save a file and return the storage path/key.

        Args:
            file: File-like object to save
            filename: Original filename
            content_type: MIME type of the file
            user_id: ID of the user uploading the file

        Returns:
            Storage path or key for retrieving the file
        """
        pass

    @abstractmethod
    async def get(self, path: str) -> Path | None:
        """
        Get the local path to a file (for serving).

        Args:
            path: Storage path returned from save()

        Returns:
            Local filesystem path or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete a file from storage.

        Args:
            path: Storage path returned from save()

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if a file exists in storage."""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.

    Files are organized by user_id and date:
    uploads/
      └── {user_id}/
          └── {year}/{month}/
              └── {unique_filename}
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate a unique filename while preserving extension."""
        # Extract extension
        ext = Path(original_filename).suffix.lower()

        # Generate unique identifier
        unique_id = uuid.uuid4().hex[:16]

        # Create timestamp prefix for sortability
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        return f"{timestamp}_{unique_id}{ext}"

    def _get_user_path(self, user_id: int) -> Path:
        """Get the storage directory for a user with date-based organization."""
        now = datetime.utcnow()
        return self.base_path / str(user_id) / str(now.year) / f"{now.month:02d}"

    async def save(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
        user_id: int,
    ) -> str:
        """Save a file to local storage."""
        # Create unique filename
        unique_filename = self._generate_unique_filename(filename)

        # Create user directory
        user_path = self._get_user_path(user_id)
        user_path.mkdir(parents=True, exist_ok=True)

        # Full file path
        file_path = user_path / unique_filename

        # Write the file
        with open(file_path, "wb") as dest:
            # Read in chunks to handle large files
            shutil.copyfileobj(file, dest)

        # Return relative path from base
        return str(file_path.relative_to(self.base_path))

    async def get(self, path: str) -> Path | None:
        """Get the local path to a file."""
        full_path = self.base_path / path
        if full_path.exists():
            return full_path
        return None

    async def delete(self, path: str) -> bool:
        """Delete a file from local storage."""
        full_path = self.base_path / path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    async def exists(self, path: str) -> bool:
        """Check if a file exists."""
        return (self.base_path / path).exists()

    def get_full_path(self, path: str) -> Path:
        """Get the full filesystem path for a storage path."""
        return self.base_path / path


class StorageService:
    """
    High-level storage service that wraps storage backends.

    Provides file validation, type checking, and storage operations.
    """

    # Allowed image MIME types
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }

    # Maximum file size (default 50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    def __init__(self, backend: StorageBackend, settings: Settings | None = None):
        self.backend = backend
        self.settings = settings or get_settings()

        # Override max file size from settings if available
        if hasattr(self.settings, "storage") and hasattr(
            self.settings.storage, "max_file_size_mb"
        ):
            self.MAX_FILE_SIZE = self.settings.storage.max_file_size_mb * 1024 * 1024

    def validate_image(self, content_type: str, file_size: int) -> tuple[bool, str]:
        """
        Validate that a file is an allowed image type and size.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if content_type not in self.ALLOWED_IMAGE_TYPES:
            return False, f"File type '{content_type}' is not allowed. Allowed types: JPEG, PNG, GIF, WebP, BMP, TIFF"

        if file_size > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File size exceeds maximum allowed ({max_mb:.0f}MB)"

        return True, ""

    async def save_file(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
        user_id: int,
    ) -> str:
        """
        Save a file to storage.

        Args:
            file: File-like object
            filename: Original filename
            content_type: MIME type
            user_id: Owner's user ID

        Returns:
            Storage path for the saved file
        """
        return await self.backend.save(file, filename, content_type, user_id)

    async def get_file_path(self, storage_path: str) -> Path | None:
        """Get the local filesystem path for serving a file."""
        return await self.backend.get(storage_path)

    async def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage."""
        return await self.backend.delete(storage_path)

    async def file_exists(self, storage_path: str) -> bool:
        """Check if a file exists."""
        return await self.backend.exists(storage_path)

    def calculate_file_hash(self, file: BinaryIO) -> str:
        """
        Calculate SHA-256 hash of a file for deduplication.

        The file position is reset after hashing.
        """
        hasher = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
        file.seek(0)  # Reset file position
        return hasher.hexdigest()


# Global storage service instance
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get the global storage service instance."""
    global _storage_service
    if _storage_service is None:
        raise RuntimeError("Storage service not initialized. Call init_storage() first.")
    return _storage_service


def init_storage(settings: Settings) -> StorageService:
    """
    Initialize the storage service.

    Should be called during application startup.
    """
    global _storage_service

    # For now, we use local storage
    # In production, you could switch to S3 based on settings
    upload_path = settings.storage.uploads_path
    backend = LocalStorageBackend(upload_path)
    _storage_service = StorageService(backend, settings)

    return _storage_service