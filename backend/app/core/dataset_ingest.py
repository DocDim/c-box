from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.artifact import ModelArtifact
from app.models.dataset import PURPOSE_EVALUATION, Dataset
from app.models.user import User
from app.core.feature_alignment import validate_dataset_columns
from app.core.storage import build_storage_path, delete_stored_file, save_upload_with_limit


@dataclass
class IngestedDataset:
    dataset: Dataset
    frame: pd.DataFrame
    validation: Dict[str, Any]


async def ingest_csv_upload(
    file: UploadFile,
    user: User,
    db: AsyncSession,
    model: Optional[ModelArtifact],
    target_column: Optional[str],
    purpose: str = PURPOSE_EVALUATION,
) -> IngestedDataset:
    """
    Stores a CSV in the user's sandbox, validates it against the model's fitted features
    (FR-2.5) and adds the Dataset row to the session (flushed, not committed). On any
    failure the stored file is removed, so callers can roll back cleanly.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only CSV datasets are supported. Received '{ext}'"
        )

    destination = build_storage_path(settings.DATASETS_DIR, user.id, file.filename, ext)
    await save_upload_with_limit(file, destination, settings.max_dataset_bytes)

    try:
        df = await run_in_threadpool(pd.read_csv, destination)
        if df.empty or len(df.columns) == 0:
            raise ValueError("CSV has no rows or columns.")
        validation = validate_dataset_columns([str(c) for c in df.columns], model, target_column or None)
    except HTTPException:
        delete_stored_file(destination)
        raise
    except Exception as e:
        delete_stored_file(destination)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {e}"
        )

    dataset = Dataset(
        owner_id=user.id,
        model_id=model.id if model else None,
        name=Path(file.filename or "dataset.csv").name[:255],
        file_path=str(destination),
        n_rows=len(df),
        columns=[str(c) for c in df.columns],
        target_column=validation["target_column"],
        purpose=purpose,
    )
    db.add(dataset)
    await db.flush()
    return IngestedDataset(dataset=dataset, frame=df, validation=validation)
