from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.models.artifact import ModelArtifact, ModelShare, AccessRole

ROLE_HIERARCHY = {
    AccessRole.VIEWER: 1,
    AccessRole.EXPLAINER: 2,
    AccessRole.ADMIN: 3
}

async def verify_model_access(
    model_id: int,
    user: User,
    db: AsyncSession,
    min_role: AccessRole = AccessRole.VIEWER
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