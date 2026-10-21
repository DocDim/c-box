import pandas as pd
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole
from app.core.dependencies import get_current_user
from app.core.permissions import verify_model_access
from app.core.model_loader import model_loader

router = APIRouter(prefix="/predict", tags=["Inference Engine"])

class PredictRequest(BaseModel):
    model_id: int
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    model_id: int
    prediction: float
    probabilities: Optional[Dict[str, float]] = None

@router.post("/", response_model=PredictResponse)
async def run_point_prediction(
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Executes single-row inference for an authorized model."""
    model_record = await verify_model_access(
        payload.model_id, current_user, db, min_role=AccessRole.VIEWER
    )
    estimator = model_loader.load_model(model_record.file_path)

    input_df = pd.DataFrame([payload.features])

    expected_dim = getattr(estimator, "n_features_in_", None)
    if expected_dim is not None and input_df.shape[1] != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dimension mismatch: Model expects {expected_dim} features, received {input_df.shape[1]}."
        )

    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(input_df)[0]
        pred_val = float(probs[1]) if len(probs) > 1 else float(probs[0])
        class_probs = {f"class_{i}": float(p) for i, p in enumerate(probs)}
        return PredictResponse(model_id=model_record.id, prediction=pred_val, probabilities=class_probs)
    else:
        pred_val = float(estimator.predict(input_df)[0])
        return PredictResponse(model_id=model_record.id, prediction=pred_val)