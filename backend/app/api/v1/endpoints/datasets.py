from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole, ModelArtifact, ModelShare
from app.models.dataset import PURPOSE_BACKGROUND
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetPreviewResponse, DatasetResponse, DatasetRowsResponse
from app.core.dependencies import get_current_user
from app.core.explainer_cache import explainer_cache
from app.core.dataset_ingest import ingest_csv_upload
from app.core.feature_alignment import frame_to_records, read_dataset_frame, summarize_frame
from app.core.permissions import verify_dataset_access, verify_model_access
from app.core.storage import delete_stored_file

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def to_dataset_response(d: Dataset, background_id: Optional[int] = None) -> DatasetResponse:
    return DatasetResponse(
        id=d.id, name=d.name, owner_id=d.owner_id, model_id=d.model_id, n_rows=d.n_rows,
        columns=d.columns, target_column=d.target_column, created_at=d.created_at,
        purpose=d.purpose, is_background=background_id is not None and background_id == d.id,
    )


@router.post("/upload", response_model=DatasetPreviewResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: Optional[int] = Form(None),
    target_column: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingests an evaluation CSV (e.g. for batch prediction), validates its columns against the
    linked model's fitted features (FR-2.5) and returns summary statistics plus a preview.
    Any user who can run the model may link one. Background data is managed separately by
    the model owner (POST /models/{id}/background) and is never created here.
    """
    model: Optional[ModelArtifact] = None
    if model_id is not None:
        model = await verify_model_access(model_id, current_user, db, min_role=AccessRole.VIEWER)

    ingested = await ingest_csv_upload(file, current_user, db, model, target_column)
    dataset, df, validation = ingested.dataset, ingested.frame, ingested.validation

    await db.commit()
    await db.refresh(dataset)

    base = to_dataset_response(dataset, model.background_dataset_id if model else None)
    return DatasetPreviewResponse(
        **base.model_dump(),
        extra_columns=validation["extra"],
        summary=summarize_frame(df),
        preview_rows=frame_to_records(df.head(5)),
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
    model_id: Optional[int] = Query(None),
    purpose: Optional[str] = Query(None, pattern="^(background|evaluation)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Datasets the user uploaded, plus those linked to models shared with them."""
    accessible_models = select(ModelArtifact.id).where(ModelArtifact.owner_id == current_user.id).union(
        select(ModelShare.model_id).where(ModelShare.user_id == current_user.id)
    )
    stmt = select(Dataset, ModelArtifact.background_dataset_id).outerjoin(
        ModelArtifact, ModelArtifact.id == Dataset.model_id
    ).where(or_(Dataset.owner_id == current_user.id, Dataset.model_id.in_(accessible_models)))
    if model_id is not None:
        stmt = stmt.where(Dataset.model_id == model_id)
    if purpose is not None:
        stmt = stmt.where(Dataset.purpose == purpose)
    rows = (await db.execute(stmt.order_by(Dataset.created_at.desc()))).all()
    return [to_dataset_response(d, bg) for d, bg in rows]


@router.get("/{dataset_id}/rows", response_model=DatasetRowsResponse)
async def get_dataset_rows(
    dataset_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Paginated row access so any record can be selected for explanation."""
    dataset = await verify_dataset_access(dataset_id, current_user, db)
    df = await run_in_threadpool(read_dataset_frame, dataset)
    page = df.iloc[offset:offset + limit]
    return DatasetRowsResponse(
        dataset_id=dataset.id, offset=offset, limit=limit, total_rows=len(df),
        columns=[str(c) for c in df.columns], rows=frame_to_records(page),
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """The uploader or the linked model's owner can delete a dataset."""
    dataset = await verify_dataset_access(dataset_id, current_user, db)
    model = await db.get(ModelArtifact, dataset.model_id) if dataset.model_id else None
    if dataset.owner_id != current_user.id and (model is None or model.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Only the uploader or model owner can delete this dataset.")
    if dataset.purpose == PURPOSE_BACKGROUND and (model is None or model.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Only the model owner can remove background data.")
    if model is not None and model.background_dataset_id == dataset.id:
        model.background_dataset_id = None
        explainer_cache.invalidate(model.id)
    file_path = dataset.file_path
    await db.delete(dataset)
    await db.commit()
    delete_stored_file(file_path)
