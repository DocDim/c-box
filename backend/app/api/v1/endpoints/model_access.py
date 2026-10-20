from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User
from app.models.artifact import ModelArtifact, ModelShare, AccessRole
from app.schemas.artifact import ShareGrantRequest, SharedUserResponse, ModelResponse
from app.core.dependencies import get_current_user
from app.core.permissions import verify_model_access

router = APIRouter(prefix="/models", tags=["Model Management & Sharing"])

@router.get("/", response_model=List[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists all models owned by the user or shared with them."""
    # Owned models
    owned_stmt = select(ModelArtifact).where(ModelArtifact.owner_id == current_user.id)
    owned = (await db.execute(owned_stmt)).scalars().all()

    # Shared models
    shared_stmt = (
        select(ModelArtifact, ModelShare.role)
        .join(ModelShare, ModelArtifact.id == ModelShare.model_id)
        .where(ModelShare.user_id == current_user.id)
    )
    shared = (await db.execute(shared_stmt)).all()

    results: List[ModelResponse] = []
    for m in owned:
        results.append(ModelResponse(
            id=m.id, name=m.name, domain_context=m.domain_context,
            estimator_type=m.estimator_type, owner_id=m.owner_id,
            is_owner=True, assigned_role="OWNER", created_at=m.created_at
        ))
    for m, role in shared:
        results.append(ModelResponse(
            id=m.id, name=m.name, domain_context=m.domain_context,
            estimator_type=m.estimator_type, owner_id=m.owner_id,
            is_owner=False, assigned_role=role.value, created_at=m.created_at
        ))
    return results

@router.post("/{model_id}/share", status_code=status.HTTP_200_OK)
async def share_model(
    model_id: int,
    payload: ShareGrantRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Grants or updates access to another user via email."""
    model = await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)

    target_user_stmt = select(User).where(User.email == payload.user_email)
    target_user = (await db.execute(target_user_stmt)).scalar_one_or_none()

    if not target_user:
        raise HTTPException(status_code=404, detail="Target user with that email does not exist.")
    if target_user.id == model.owner_id:
        raise HTTPException(status_code=400, detail="Cannot share with the model owner.")

    share_stmt = select(ModelShare).where(
        ModelShare.model_id == model_id,
        ModelShare.user_id == target_user.id
    )
    existing_share = (await db.execute(share_stmt)).scalar_one_or_none()

    if existing_share:
        existing_share.role = payload.role
    else:
        new_share = ModelShare(model_id=model_id, user_id=target_user.id, role=payload.role)
        db.add(new_share)

    await db.commit()
    return {"message": f"Successfully granted {payload.role.value} access to {payload.user_email}."}