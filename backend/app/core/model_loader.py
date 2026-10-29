import multiprocessing
from pathlib import Path
from typing import Any, Dict
from fastapi import HTTPException, status
from app.config import settings
from app.core.storage import ensure_within_storage
from app.core import _inspect_worker

class ModelLoaderService:
    @staticmethod
    def validate_file_extension(filename: str) -> str:
        ext = Path(filename or "").suffix.lower()
        allowed = settings.allowed_model_extensions()
        if ext == ".h5":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Keras .h5 models are out of scope for v1.0 (tabular scikit-learn compatible estimators only)."
            )
        if ext not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Allowed: {sorted(allowed)}"
            )
        return ext

    @staticmethod
    def file_format_for(ext: str) -> str:
        return "skops" if ext == ".skops" else "joblib"

    @staticmethod
    def inspect_in_subprocess(file_path: Path, file_format: str) -> Dict[str, Any]:
        """
        Deserializes and inspects an uploaded artifact in a disposable child process (NFR-1).
        Hangs are killed after MODEL_INSPECTION_TIMEOUT_SECONDS; crashes never take the API down.
        Note: a child process isolates failures, not privileges. Pickle-based formats can still
        execute code as the server's OS user, which is why ALLOW_PICKLE_MODELS exists.
        """
        real_path = ensure_within_storage(file_path)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=1)
        try:
            job = pool.apply_async(_inspect_worker.inspect_artifact, (str(real_path), file_format))
            return job.get(timeout=settings.MODEL_INSPECTION_TIMEOUT_SECONDS)
        except multiprocessing.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Model inspection timed out; the artifact was rejected."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to deserialize model binary: {e}"
            )
        finally:
            pool.terminate()
            pool.join()

    @staticmethod
    def load_model(file_path: str, file_format: str = "joblib") -> Any:
        """Loads an already-validated artifact in-process for inference / SHAP."""
        real_path = ensure_within_storage(file_path)
        if not real_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model file not found on disk."
            )
        try:
            return _inspect_worker.load_artifact(str(real_path), file_format)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to deserialize model binary: {e}"
            )

    @staticmethod
    def inspect_model(model: Any) -> Dict[str, Any]:
        return _inspect_worker.inspect_model(model)

model_loader = ModelLoaderService()
