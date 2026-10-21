import os
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole
from app.schemas.explain import (
    InstanceExplainRequest,
    InstanceExplainResponse,
    GlobalExplainResponse
)
from app.core.dependencies import get_current_user
from app.core.permissions import verify_model_access
from app.core.model_loader import model_loader
from app.core.shap_engine import ShapExplainerEngine

router = APIRouter(prefix="/explain", tags=["Explainability Engine"])

@router.post("/instance", response_model=InstanceExplainResponse)
async def explain_instance(
    payload: InstanceExplainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generates local SHAP feature attributions for a single instance."""
    model_record = await verify_model_access(
        payload.model_id, current_user, db, min_role=AccessRole.EXPLAINER
    )
    estimator = model_loader.load_model(model_record.file_path)
    instance_df = pd.DataFrame([payload.instance_features])

    try:
        engine = ShapExplainerEngine(model=estimator, background_data=instance_df)
        result = engine.explain_instance(instance_df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation failed: {str(e)}"
        )

    return InstanceExplainResponse(
        model_id=model_record.id,
        prediction=result["prediction"],
        base_value=result["base_value"],
        feature_contributions=result["feature_contributions"]
    )

@router.get("/global/{model_id}", response_model=GlobalExplainResponse)
async def explain_global(
    model_id: int,
    dataset_filename: str = Query(..., description="CSV filename in user's dataset storage"),
    max_samples: int = Query(200, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Computes global feature importance across an uploaded evaluation CSV."""
    model_record = await verify_model_access(
        model_id, current_user, db, min_role=AccessRole.EXPLAINER
    )
    estimator = model_loader.load_model(model_record.file_path)

    dataset_path = settings.DATASETS_DIR / str(current_user.id) / dataset_filename
    real_path = os.path.realpath(dataset_path)
    
    if not real_path.startswith(os.path.realpath(settings.STORAGE_DIR)) or not os.path.exists(real_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file not found.")

    df = pd.read_csv(real_path)

    try:
        engine = ShapExplainerEngine(model=estimator, background_data=df.head(50))
        result = engine.explain_global(df, max_samples=max_samples)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global SHAP explanation failed: {str(e)}"
        )

    return GlobalExplainResponse(
        model_id=model_record.id,
        samples_evaluated=result["samples_evaluated"],
        global_importance=result["global_importance"]
    )