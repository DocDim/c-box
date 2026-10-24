from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

# A background dataset is the model's SHAP reference distribution and only feeds explanations
# and global insights; evaluation datasets are the ones users run predictions on.
PURPOSE_BACKGROUND = "background"
PURPOSE_EVALUATION = "evaluation"

if TYPE_CHECKING:
    from app.models.artifact import ModelArtifact


class Dataset(Base):
    """Evaluation / reference CSV registered against a model (FR-2.5)."""
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Datasets linked to a model are readable by everyone the model is shared with
    model_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    target_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    purpose: Mapped[str] = mapped_column(String(16), default=PURPOSE_EVALUATION, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    model: Mapped[Optional["ModelArtifact"]] = relationship(
        "ModelArtifact", back_populates="datasets", foreign_keys=[model_id]
    )
