from datetime import datetime, timezone
from typing import List, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.artifact import ModelArtifact, ModelShare

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    owned_models: Mapped[List["ModelArtifact"]] = relationship(
        "ModelArtifact", back_populates="owner", cascade="all, delete-orphan"
    )
    shared_models: Mapped[List["ModelShare"]] = relationship(
        "ModelShare", back_populates="user", cascade="all, delete-orphan"
    )