import re
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile, status
from app.config import settings

_CHUNK_SIZE = 1024 * 1024
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Strips directory components and unsafe characters from a client-supplied name."""
    stem = Path(filename or "upload").stem
    cleaned = _UNSAFE_CHARS.sub("_", stem).strip("._")
    return (cleaned or "upload")[:100]


def ensure_within_storage(path: Path | str) -> Path:
    """
    Resolves symlinks / '..' segments and verifies the result stays inside STORAGE_DIR.
    Uses Path.is_relative_to rather than a string prefix check, so sibling folders
    such as 'storage_evil' are rejected.
    """
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(settings.STORAGE_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security violation: Invalid file path access."
        )
    return resolved


def build_storage_path(base_dir: Path, user_id: int, original_filename: str, ext: str) -> Path:
    """Returns storage/<kind>/<user_id>/<uuid>_<sanitized_name><ext> (SDD 2.2)."""
    user_dir = base_dir / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    destination = user_dir / f"{uuid.uuid4().hex}_{sanitize_filename(original_filename)}{ext}"
    return ensure_within_storage(destination)


async def save_upload_with_limit(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Streams an upload to disk, aborting (and deleting the partial file) past max_bytes."""
    written = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await upload.read(_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
                    )
                buffer.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    return written


def delete_stored_file(path: Path | str) -> None:
    """Deletes a stored artifact if it lives inside the sandbox; silently ignores missing files."""
    try:
        ensure_within_storage(path).unlink(missing_ok=True)
    except HTTPException:
        pass
