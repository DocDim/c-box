import csv
import io
import time
from typing import Any, Dict, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole, ModelArtifact
from app.models.dataset import Dataset
from app.schemas.explain import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictionRow,
    PredictRequest,
    PredictResponse,
)
from app.core.dependencies import get_current_user
from app.core.feature_alignment import align_frame, frame_to_records, read_dataset_frame
from app.core.inference import predict_batch, predict_frame
from app.core.instances import effective_task_type, get_estimator, resolve_instance
from app.core.permissions import ensure_not_background, verify_dataset_access, verify_model_access
from app.core.storage import sanitize_filename

router = APIRouter(prefix="/predict", tags=["Inference Engine"])


@router.post("/", response_model=PredictResponse)
async def run_point_prediction(
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Executes single-row inference for an authorized model (VIEWER and above)."""
    model_record = await verify_model_access(
        payload.model_id, current_user, db, min_role=AccessRole.VIEWER
    )
    instance_df, input_features = await resolve_instance(payload, model_record, current_user, db)
    estimator = await get_estimator(model_record)
    task_type = effective_task_type(model_record, estimator)
    result = await run_in_threadpool(predict_frame, estimator, task_type, instance_df)
    return PredictResponse(model_id=model_record.id, input_features=input_features, **result)


async def _run_batch(
    model_id: int, dataset_id: int, user: User, db: AsyncSession
) -> Tuple[ModelArtifact, Dataset, pd.DataFrame, pd.DataFrame, Dict[str, Any], float]:
    """Aligns the whole dataset to the model and predicts every row in one vectorized call."""
    model = await verify_model_access(model_id, user, db, min_role=AccessRole.VIEWER)
    dataset = await verify_dataset_access(dataset_id, user, db)
    ensure_not_background(dataset)
    estimator = await get_estimator(model)
    task_type = effective_task_type(model, estimator)

    def _work():
        raw = read_dataset_frame(dataset)
        features = align_frame(raw, model, dataset.target_column)
        actual = raw[dataset.target_column] if dataset.target_column in raw.columns else None
        started = time.perf_counter()
        result = predict_batch(estimator, task_type, features, actual)
        return raw, features, result, (time.perf_counter() - started) * 1000

    raw, features, result, elapsed = await run_in_threadpool(_work)
    return model, dataset, raw, features, result, elapsed


@router.post("/batch", response_model=BatchPredictResponse)
async def run_batch_prediction(
    payload: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Predicts every row of a dataset (VIEWER and above) and returns one page of rows plus
    dataset-wide class counts / prediction statistics and, when a target column exists,
    accuracy or regression error metrics.
    """
    model, dataset, raw, features, result, elapsed = await _run_batch(
        payload.model_id, payload.dataset_id, current_user, db
    )
    end = min(payload.offset + payload.limit, len(features))
    page_features = frame_to_records(features.iloc[payload.offset:end])
    actual_col = dataset.target_column if dataset.target_column in raw.columns else None
    actual_values = frame_to_records(raw[[actual_col]].iloc[payload.offset:end]) if actual_col else None

    rows = []
    for i, feats in enumerate(page_features):
        idx = payload.offset + i
        rows.append(BatchPredictionRow(
            row_index=idx,
            features=feats,
            prediction=result["prediction"][idx],
            predicted_class=result["predicted_class"][idx] if result.get("predicted_class") else None,
            probabilities=result["probabilities"][idx] if result.get("probabilities") else None,
            actual=actual_values[i][actual_col] if actual_values else None,
        ))

    return BatchPredictResponse(
        model_id=model.id,
        dataset_id=dataset.id,
        task_type=result["task_type"],
        total_rows=len(features),
        offset=payload.offset,
        limit=payload.limit,
        target_column=actual_col,
        class_counts=result["class_counts"],
        prediction_stats=result["prediction_stats"],
        metrics=result["metrics"],
        predict_ms=elapsed,
        rows=rows,
    )


@router.get("/batch/export.csv")
async def export_batch_prediction(
    model_id: int = Query(...),
    dataset_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """The original dataset with predicted class / value and per-class probabilities appended."""
    model, dataset, raw, _features, result, _ = await _run_batch(model_id, dataset_id, current_user, db)

    out = raw.copy()
    if result.get("predicted_class"):
        out["predicted_class"] = result["predicted_class"]
    if any(p is not None for p in result["prediction"]):
        out["prediction" if not result.get("probabilities") else "predicted_probability"] = result["prediction"]
    for label in result.get("labels", []):
        out[f"prob_{label}"] = [p[label] for p in result["probabilities"]]

    buffer = io.StringIO()
    out.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    stem = sanitize_filename(dataset.name)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_predictions_model{model.id}.csv"'},
    )
