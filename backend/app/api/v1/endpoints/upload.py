import uuid
import shutil
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.models.artifact import ModelArtifact
from app.schemas.artifact import ModelResponse
from app.schemas.dataset import DatasetPreviewResponse
from app.core.dependencies import get_current_user
from app.core.model_loader import model_loader

router = APIRouter(prefix="/upload", tags=["Upload & Ingestion"])

@router.post("/model", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    name: str = Form(...),
    domain_context: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Ingests trained model binary, validates serialization, and registers metadata."""
    ext = model_loader.validate_file_extension(file.filename)
    
    # User-isolated filesystem storage
    user_dir = settings.MODELS_DIR / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    saved_filename = f"{uuid.uuid4().hex}_{Path(file.filename).stem}{ext}"
    destination = user_dir / saved_filename

    # Save to disk
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Inspect model binary
    loaded_model = model_loader.load_model(str(destination))
    inspection = model_loader.inspect_model(loaded_model)

    # Persist in SQLite
    artifact = ModelArtifact(
        name=name,
        domain_context=domain_context,
        estimator_type=inspection["estimator_type"],
        file_path=str(destination),
        owner_id=current_user.id
    )
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)

    return ModelResponse(
        id=artifact.id,
        name=artifact.name,
        domain_context=artifact.domain_context,
        estimator_type=artifact.estimator_type,
        owner_id=artifact.owner_id,
        is_owner=True,
        assigned_role="OWNER",
        created_at=artifact.created_at
    )

@router.post("/dataset", response_model=DatasetPreviewResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Ingests CSV dataset and returns structural column preview."""
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only CSV datasets are supported. Received '{ext}'"
        )

    user_dir = settings.DATASETS_DIR / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    saved_filename = f"{uuid.uuid4().hex}_{file.filename}"
    destination = user_dir / saved_filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(destination)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {str(e)}"
        )

    preview_records = df.head(5).fillna("").to_dict(orient="records")

    return DatasetPreviewResponse(
        filename=saved_filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=list(df.columns),
        preview_rows=preview_records
    )