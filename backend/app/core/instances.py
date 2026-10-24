from typing import Any, Dict, Optional, Tuple

import pandas as pd
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import ModelArtifact, TaskType
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.explain import InstanceSource
from app.core.explainer_cache import explainer_cache
from app.core.feature_alignment import align_frame, align_instance, frame_to_records, read_dataset_frame
from app.core.model_loader import model_loader
from app.core.permissions import ensure_not_background, verify_dataset_access
from app.core.shap_engine import BackgroundDataRequired, ShapExplainerEngine


async def resolve_instance(
    payload: InstanceSource, model: ModelArtifact, user: User, db: AsyncSession
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Turns manual inputs or a (dataset_id, row_index) reference into an aligned 1-row frame."""
    if payload.dataset_id is not None and payload.row_index is not None:
        dataset = await verify_dataset_access(payload.dataset_id, user, db)
        ensure_not_background(dataset)
        if payload.row_index >= dataset.n_rows:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Row index {payload.row_index} is out of range (dataset has {dataset.n_rows} rows)."
            )
        frame = await run_in_threadpool(read_dataset_frame, dataset)
        row = frame.iloc[[payload.row_index]].reset_index(drop=True)
        # Manual overrides on top of a dataset row ("what-if" edits)
        if payload.instance_features:
            for key, value in payload.instance_features.items():
                row[key] = value
        instance = align_frame(row, model, dataset.target_column)
    else:
        instance = align_instance(payload.instance_features or {}, model)
    return instance, frame_to_records(instance)[0]


def effective_task_type(model: ModelArtifact, estimator: Any) -> str:
    if model.task_type != TaskType.UNKNOWN.value:
        return model.task_type
    return model_loader.inspect_model(estimator)["task_type"]


def _load_estimator(model: ModelArtifact) -> Any:
    return explainer_cache.get_estimator(
        model.id, model.file_path, lambda: model_loader.load_model(model.file_path, model.file_format)
    )


async def get_estimator(model: ModelArtifact) -> Any:
    return await run_in_threadpool(_load_estimator, model)


async def get_engine(model: ModelArtifact, db: AsyncSession) -> ShapExplainerEngine:
    """Returns a cached SHAP engine whose background is the model's reference dataset."""
    background_ds: Optional[Dataset] = None
    if model.background_dataset_id is not None:
        background_ds = await db.get(Dataset, model.background_dataset_id)

    def build():
        estimator = _load_estimator(model)
        background = None
        if background_ds is not None:
            background = align_frame(read_dataset_frame(background_ds), model, background_ds.target_column)
        return estimator, effective_task_type(model, estimator), background

    try:
        return await run_in_threadpool(
            explainer_cache.get_or_create,
            model.id,
            model.file_path,
            background_ds.id if background_ds else None,
            build,
        )
    except BackgroundDataRequired as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
