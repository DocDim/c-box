from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.models.artifact import ModelArtifact, ModelShare, AccessRole
from app.models.dataset import PURPOSE_BACKGROUND, Dataset
from app.schemas.artifact import ModelResponse, ShareGrantRequest, SharedUserResponse
from app.core.dependencies import get_current_user
from app.core.explainer_cache import explainer_cache
from app.core.dataset_ingest import ingest_csv_upload
from app.core.model_loader import model_loader
from app.core.permissions import OWNER_ROLE, get_effective_role, verify_model_access
from app.core.storage import build_storage_path, delete_stored_file, save_upload_with_limit

router = APIRouter(prefix="/models", tags=["Model Management & Sharing"])


def to_model_response(m: ModelArtifact, role: str, owner_email: Optional[str] = None) -> ModelResponse:
    return ModelResponse(
        id=m.id, name=m.name, domain_context=m.domain_context,
        estimator_type=m.estimator_type, owner_id=m.owner_id, owner_email=owner_email,
        is_owner=role == OWNER_ROLE, assigned_role=role, created_at=m.created_at,
        task_type=m.task_type, file_format=m.file_format, n_features=m.n_features,
        feature_names=m.feature_names, class_labels=m.class_labels,
        background_dataset_id=m.background_dataset_id,
    )


@router.post("/upload", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    name: str = Form(..., min_length=1, max_length=255),
    domain_context: str = Form(..., min_length=1, max_length=255),
    file: UploadFile = File(...),
    background_file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingests a trained model, inspects it in a sandboxed child process and registers metadata.
    An optional background CSV is validated against the model's features and registered as its
    SHAP reference distribution in the same transaction: if it is rejected, nothing is saved.
    """
    ext = model_loader.validate_file_extension(file.filename)
    file_format = model_loader.file_format_for(ext)
    destination = build_storage_path(settings.MODELS_DIR, current_user.id, file.filename, ext)
    await save_upload_with_limit(file, destination, settings.max_model_bytes)

    try:
        inspection = await run_in_threadpool(model_loader.inspect_in_subprocess, destination, file_format)
    except HTTPException:
        delete_stored_file(destination)
        raise

    artifact = ModelArtifact(
        name=name.strip(),
        domain_context=domain_context.strip(),
        estimator_type=inspection["estimator_type"],
        task_type=inspection["task_type"],
        n_features=inspection["n_features"],
        feature_names=inspection["feature_names"],
        class_labels=inspection["class_labels"],
        file_path=str(destination),
        file_format=file_format,
        owner_id=current_user.id
    )
    db.add(artifact)
    await db.flush()

    if background_file is not None and background_file.filename:
        try:
            ingested = await ingest_csv_upload(
                background_file, current_user, db, artifact, target_column, purpose=PURPOSE_BACKGROUND
            )
        except HTTPException as e:
            await db.rollback()
            delete_stored_file(destination)
            raise HTTPException(status_code=e.status_code, detail=f"Background dataset rejected: {e.detail}")
        artifact.background_dataset_id = ingested.dataset.id

    await db.commit()
    await db.refresh(artifact)
    return to_model_response(artifact, OWNER_ROLE, current_user.email)


@router.get("/", response_model=List[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists all models owned by the user or shared with them."""
    owned = (await db.execute(
        select(ModelArtifact).where(ModelArtifact.owner_id == current_user.id)
        .order_by(ModelArtifact.created_at.desc())
    )).scalars().all()

    shared = (await db.execute(
        select(ModelArtifact, ModelShare.role, User.email)
        .join(ModelShare, ModelArtifact.id == ModelShare.model_id)
        .join(User, User.id == ModelArtifact.owner_id)
        .where(ModelShare.user_id == current_user.id)
        .order_by(ModelArtifact.created_at.desc())
    )).all()

    results = [to_model_response(m, OWNER_ROLE, current_user.email) for m in owned]
    results += [to_model_response(m, role.value, email) for m, role, email in shared]
    return results


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    model = await verify_model_access(model_id, current_user, db, min_role=AccessRole.VIEWER)
    role = await get_effective_role(model, current_user, db)
    owner = await db.get(User, model.owner_id)
    return to_model_response(model, role or AccessRole.VIEWER.value, owner.email if owner else None)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Owner-only: removes the artifact, its shares, its explanation history and its background
    data (which has no use without the model). Evaluation datasets are kept but unlinked.
    """
    model = await verify_model_access(model_id, current_user, db, require_owner=True)
    file_paths = [model.file_path]
    backgrounds = (await db.execute(select(Dataset).where(
        Dataset.model_id == model_id, Dataset.purpose == PURPOSE_BACKGROUND
    ))).scalars().all()
    model.background_dataset_id = None
    await db.flush()
    for dataset in backgrounds:
        file_paths.append(dataset.file_path)
        await db.delete(dataset)
    await db.delete(model)
    await db.commit()
    explainer_cache.invalidate(model_id)
    for path in file_paths:
        delete_stored_file(path)


@router.post("/{model_id}/background", response_model=ModelResponse)
async def replace_background_dataset(
    model_id: int,
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Owner-only: uploads (or replaces) the model's background data, the SHAP reference
    distribution used for explanations and global insights. Predictions never run on it.
    The previous background dataset and its file are removed.
    """
    model = await verify_model_access(model_id, current_user, db, require_owner=True)
    previous = await db.get(Dataset, model.background_dataset_id) if model.background_dataset_id else None

    ingested = await ingest_csv_upload(file, current_user, db, model, target_column, purpose=PURPOSE_BACKGROUND)
    model.background_dataset_id = ingested.dataset.id

    stale_path = None
    if previous is not None and previous.purpose == PURPOSE_BACKGROUND:
        stale_path = previous.file_path
        await db.delete(previous)
    await db.commit()
    await db.refresh(model)
    explainer_cache.invalidate(model_id)
    if stale_path:
        delete_stored_file(stale_path)
    return to_model_response(model, OWNER_ROLE, current_user.email)


@router.get("/{model_id}/shares", response_model=List[SharedUserResponse])
async def list_shares(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    rows = (await db.execute(
        select(ModelShare, User.email)
        .join(User, User.id == ModelShare.user_id)
        .where(ModelShare.model_id == model_id)
        .order_by(ModelShare.granted_at)
    )).all()
    return [
        SharedUserResponse(user_id=s.user_id, email=email, role=s.role, granted_at=s.granted_at)
        for s, email in rows
    ]


@router.post("/{model_id}/share", response_model=SharedUserResponse)
async def share_model(
    model_id: int,
    payload: ShareGrantRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Grants or updates access to another user via email. Only the owner may grant ADMIN."""
    model = await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    is_owner = model.owner_id == current_user.id

    if payload.role == AccessRole.ADMIN and not is_owner:
        raise HTTPException(status_code=403, detail="Only the model owner can grant ADMIN access.")

    target_user = (await db.execute(
        select(User).where(User.email == payload.user_email)
    )).scalar_one_or_none()
    if not target_user:
        raise HTTPException(status_code=404, detail="Target user with that email does not exist.")
    if target_user.id == model.owner_id:
        raise HTTPException(status_code=400, detail="Cannot share with the model owner.")
    if target_user.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot change your own access level.")

    share = (await db.execute(select(ModelShare).where(
        ModelShare.model_id == model_id, ModelShare.user_id == target_user.id
    ))).scalar_one_or_none()

    if share:
        if share.role == AccessRole.ADMIN and not is_owner:
            raise HTTPException(status_code=403, detail="Only the model owner can modify an ADMIN's access.")
        share.role = payload.role
    else:
        share = ModelShare(model_id=model_id, user_id=target_user.id, role=payload.role)
        db.add(share)

    await db.commit()
    await db.refresh(share)
    return SharedUserResponse(
        user_id=target_user.id, email=target_user.email, role=share.role, granted_at=share.granted_at
    )


@router.delete("/{model_id}/share/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_share(
    model_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revokes a delegated grant. Owners may revoke anyone; ADMINs only non-ADMIN grants."""
    model = await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    share = (await db.execute(select(ModelShare).where(
        ModelShare.model_id == model_id, ModelShare.user_id == user_id
    ))).scalar_one_or_none()
    if not share:
        raise HTTPException(status_code=404, detail="No share exists for that user.")
    if share.role == AccessRole.ADMIN and model.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the model owner can revoke ADMIN access.")
    await db.delete(share)
    await db.commit()
