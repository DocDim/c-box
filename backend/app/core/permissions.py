from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.models.artifact import ModelArtifact, ModelShare, AccessRole
from app.models.dataset import PURPOSE_BACKGROUND, Dataset

ROLE_HIERARCHY = {
    AccessRole.VIEWER: 1,
    AccessRole.EXPLAINER: 2,
    AccessRole.ADMIN: 3
}

# Sentinel role label used in API responses for model owners
OWNER_ROLE = "OWNER"


async def get_effective_role(model: ModelArtifact, user: User, db: AsyncSession) -> Optional[str]:
    """Returns 'OWNER', a delegated AccessRole value, or None when the user has no access."""
    if model.owner_id == user.id:
        return OWNER_ROLE
    share_stmt = select(ModelShare).where(
        ModelShare.model_id == model.id,
        ModelShare.user_id == user.id
    )
    share = (await db.execute(share_stmt)).scalar_one_or_none()
    return share.role.value if share else None


async def verify_model_access(
    model_id: int,
    user: User,
    db: AsyncSession,
    min_role: AccessRole = AccessRole.VIEWER,
    require_owner: bool = False,
) -> ModelArtifact:
    """
    Validates whether the user is the direct owner or holds sufficient delegated access.
    Returns the ModelArtifact or raises HTTP 403/404.
    """
    # 1. Fetch Model Artifact
    stmt = select(ModelArtifact).where(ModelArtifact.id == model_id)
    model = (await db.execute(stmt)).scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact with ID {model_id} was not found."
        )

    # 2. Direct Ownership Bypass (Owners have full permissions)
    if model.owner_id == user.id:
        return model

    if require_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the model owner can perform this action."
        )

    # 3. Delegated Share Verification
    share_stmt = select(ModelShare).where(
        ModelShare.model_id == model_id,
        ModelShare.user_id == user.id
    )
    share = (await db.execute(share_stmt)).scalar_one_or_none()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permissions for this model."
        )

    # 4. Role Hierarchy Check
    if ROLE_HIERARCHY[share.role] < ROLE_HIERARCHY[min_role]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions: Requires '{min_role.value}' role."
        )

    return model


async def verify_dataset_access(dataset_id: int, user: User, db: AsyncSession) -> Dataset:
    """
    A dataset is readable by its uploader, or by anyone with at least VIEWER
    access to the model it is linked to.
    """
    dataset = (await db.execute(select(Dataset).where(Dataset.id == dataset_id))).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    if dataset.owner_id == user.id:
        return dataset
    if dataset.model_id is not None:
        await verify_model_access(dataset.model_id, user, db, min_role=AccessRole.VIEWER)
        return dataset
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for this dataset.")


def ensure_not_background(dataset: Dataset) -> None:
    """Background data is a SHAP reference distribution only; predictions never run on it."""
    if dataset.purpose == PURPOSE_BACKGROUND:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This is the model's background data, reserved for SHAP reference and global insights. "
                   "Load an evaluation dataset to run predictions."
        )
