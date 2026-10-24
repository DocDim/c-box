import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class ExplanationRecord(Base):
    """
    Server-side record of one local SHAP explanation. The LLM is only ever grounded
    on these stored values, never on client-supplied attributions. The table doubles
    as the explanation audit trail for compliance review.
    """
    __tablename__ = "explanation_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    row_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    explainer_type: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_class: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    explained_class: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    probabilities: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    base_value: Mapped[float] = mapped_column(Float, nullable=False)
    explained_output: Mapped[float] = mapped_column(Float, nullable=False)
    output_units: Mapped[str] = mapped_column(String(32), nullable=False)
    additivity_gap: Mapped[float] = mapped_column(Float, nullable=False)
    feature_contributions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)

    shap_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="explanation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    """One turn of the grounded conversation attached to an explanation (FR-4.4)."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    explanation_id: Mapped[str] = mapped_column(
        ForeignKey("explanation_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # "narrative" | "followup"
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Assistant-only telemetry used for the evaluation report
    llm_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    explanation: Mapped["ExplanationRecord"] = relationship("ExplanationRecord", back_populates="messages")
