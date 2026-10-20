import os
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
from app.config import settings

class ModelLoaderService:
    @staticmethod
    def validate_file_extension(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Allowed: {settings.ALLOWED_MODEL_EXTENSIONS}"
            )
        return ext

    @staticmethod
    def load_model(file_path: str) -> Any:
        # Prevent path traversal attacks
        real_path = os.path.realpath(file_path)
        real_storage = os.path.realpath(settings.STORAGE_DIR)
        if not real_path.startswith(real_storage):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Security violation: Invalid file path access."
            )
        if not os.path.exists(real_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model file not found on disk."
            )
        try:
            return joblib.load(real_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to deserialize model binary: {str(e)}"
            )

    @staticmethod
    def inspect_model(model: Any) -> Dict[str, Any]:
        """Extracts estimator architecture and expected input feature dimensions."""
        estimator_type = type(model).__name__
        n_features = getattr(model, "n_features_in_", None)
        feature_names: Optional[List[str]] = None

        if hasattr(model, "feature_names_in_"):
            feature_names = [str(f) for f in model.feature_names_in_]

        return {
            "estimator_type": estimator_type,
            "n_features_in": n_features,
            "feature_names": feature_names
        }

model_loader = ModelLoaderService()