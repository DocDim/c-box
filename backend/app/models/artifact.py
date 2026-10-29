import enum
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, ForeignKey, Enum, UniqueConstraint, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.dataset import Dataset

class AccessRole(str, enum.Enum):
    VIEWER = "VIEWER"
    EXPLAINER = "EXPLAINER"
    ADMIN = "ADMIN"

class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain_context: Mapped[str] = mapped_column(String(255), nullable=False)
    estimator_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_format: Mapped[str] = mapped_column(String(16), default="joblib", nullable=False)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Introspection results recorded at ingestion (FR-2.2)
    n_features: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feature_names: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    task_type: Mapped[str] = mapped_column(String(32), default=TaskType.UNKNOWN.value, nullable=False)
    class_labels: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    # Business metadata the prompt layer uses to speak the reader's language
    feature_glossary: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)
    outcome_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    class_descriptions: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)
    decision_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Reference data used as the SHAP background distribution (FR-3.2)
    background_dataset_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL", use_alter=True), nullable=True
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="owned_models")
    shares: Mapped[List["ModelShare"]] = relationship(
        "ModelShare", back_populates="model", cascade="all, delete-orphan", passive_deletes=True
    )
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset", back_populates="model", foreign_keys="Dataset.model_id", passive_deletes=True
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
