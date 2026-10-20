import enum
from datetime import datetime, timezone
from typing import List, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, ForeignKey, Enum, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User

class AccessRole(str, enum.Enum):
    VIEWER = "VIEWER"
    EXPLAINER = "EXPLAINER"
    ADMIN = "ADMIN"

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain_context: Mapped[str] = mapped_column(String(255), nullable=False)
    estimator_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="owned_models")
    shares: Mapped[List["ModelShare"]] = relationship(
        "ModelShare", back_populates="model", cascade="all, delete-orphan"
    )

class ModelShare(Base):
    __tablename__ = "model_shares"
    __table_args__ = (
        UniqueConstraint("model_id", "user_id", name="uq_model_user_share"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[AccessRole] = mapped_column(
        Enum(AccessRole), default=AccessRole.EXPLAINER, nullable=False
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    model: Mapped["ModelArtifact"] = relationship("ModelArtifact", back_populates="shares")
    user: Mapped["User"] = relationship("User", back_populates="shared_models")