import csv
import io
import json
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole
from app.models.explanation import ChatMessage, ExplanationRecord
from app.schemas.audit import AuditExplanation, AuditListResponse, AuditMetricsResponse, LatencyStats
from app.schemas.chat import ChatMessageResponse
from app.core.dependencies import get_current_user
from app.core.permissions import verify_model_access
from app.core.prompt_builder import DEFAULT_AUDIENCE

router = APIRouter(prefix="/audit", tags=["Audit & Evaluation"])


def _latency(values: List[Optional[float]]) -> LatencyStats:
    clean = np.array([v for v in values if v is not None], dtype=float)
    if clean.size == 0:
        return LatencyStats(count=0)
    return LatencyStats(
        count=int(clean.size),
        mean=float(clean.mean()),
        p50=float(np.percentile(clean, 50)),
        p95=float(np.percentile(clean, 95)),
        max=float(clean.max()),
    )


def _rate(flags: List[bool]) -> Optional[float]:
    return float(np.mean(flags)) if flags else None


@router.get("/models/{model_id}/explanations", response_model=AuditListResponse)
async def list_explanations(
    model_id: int,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Explanation audit trail (who explained what, with which inputs and narratives). ADMIN / OWNER."""
    await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    total = (await db.execute(
        select(func.count()).select_from(ExplanationRecord).where(ExplanationRecord.model_id == model_id)
    )).scalar_one()
    rows = (await db.execute(
        select(ExplanationRecord, User.email)
        .join(User, User.id == ExplanationRecord.user_id)
        .where(ExplanationRecord.model_id == model_id)
        .options(selectinload(ExplanationRecord.messages))
        .order_by(ExplanationRecord.created_at.desc())
        .offset(offset).limit(limit)
    )).all()

    items = [
        AuditExplanation(
            explanation_id=r.id, user_email=email, created_at=r.created_at,
            explainer_type=r.explainer_type, task_type=r.task_type, predicted_class=r.predicted_class,
            prediction=r.prediction, base_value=r.base_value, output_units=r.output_units,
            additivity_gap=r.additivity_gap, shap_ms=r.shap_ms, input_hash=r.input_hash,
            input_features=r.input_features, dataset_id=r.dataset_id, row_index=r.row_index,
            feature_contributions=r.feature_contributions,
            messages=[ChatMessageResponse.model_validate(m) for m in r.messages],
        )
        for r, email in rows
    ]
    return AuditListResponse(model_id=model_id, total=total, items=items)


@router.get("/models/{model_id}/metrics", response_model=AuditMetricsResponse)
async def model_metrics(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Aggregated performance and faithfulness metrics for the evaluation report."""
    await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    records = (await db.execute(
        select(ExplanationRecord.shap_ms, ExplanationRecord.additivity_gap)
        .where(ExplanationRecord.model_id == model_id)
    )).all()
    answers = (await db.execute(
        select(ChatMessage)
        .join(ExplanationRecord, ExplanationRecord.id == ChatMessage.explanation_id)
        .where(ExplanationRecord.model_id == model_id, ChatMessage.role == "assistant")
    )).scalars().all()

    faith = [m.faithfulness or {} for m in answers]
    narratives = [m for m in answers if m.kind == "narrative"]
    checked = sum(f.get("numbers_checked", 0) for f in faith)
    verified = sum(f.get("numbers_verified", 0) for f in faith)

    return AuditMetricsResponse(
        model_id=model_id,
        explanations=len(records),
        llm_responses=len(answers),
        narratives=len(narratives),
        followups=len(answers) - len(narratives),
        shap_ms=_latency([r.shap_ms for r in records]),
        llm_latency_ms=_latency([m.latency_ms for m in answers]),
        llm_ttft_ms=_latency([m.ttft_ms for m in answers]),
        max_additivity_gap=max((r.additivity_gap for r in records), default=None),
        faithfulness_zero_fabrication_rate=_rate([not f.get("fabricated", False) for f in faith]),
        faithfulness_numbers_verified_rate=(verified / checked) if checked else None,
        narrative_structure_complete_rate=_rate(
            [bool((m.faithfulness or {}).get("structure_complete")) for m in narratives]
        ),
        narrative_direction_error_rate=_rate(
            [bool((m.faithfulness or {}).get("direction_errors")) for m in narratives]
        ),
        mean_features_mentioned=(
            float(np.mean([len(f.get("features_mentioned", [])) for f in faith])) if faith else None
        ),
    )


@router.get("/models/{model_id}/export.csv")
async def export_csv(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """One row per LLM answer (or per explanation without answers) for offline analysis."""
    await verify_model_access(model_id, current_user, db, min_role=AccessRole.ADMIN)
    rows = (await db.execute(
        select(ExplanationRecord, User.email)
        .join(User, User.id == ExplanationRecord.user_id)
        .where(ExplanationRecord.model_id == model_id)
        .options(selectinload(ExplanationRecord.messages))
        .order_by(ExplanationRecord.created_at)
    )).all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "explanation_id", "created_at", "user_email", "explainer_type", "task_type", "predicted_class",
        "prediction", "base_value", "output_units", "additivity_gap", "shap_ms", "input_hash",
        "message_kind", "audience", "question", "llm_model", "llm_latency_ms", "llm_ttft_ms",
        "numbers_checked", "numbers_verified", "fabricated", "structure_complete", "direction_errors",
        "misplaced_features", "features_mentioned", "answer",
    ])
    for record, email in rows:
        base = [
            record.id, record.created_at.isoformat(), email, record.explainer_type, record.task_type,
            record.predicted_class, record.prediction, record.base_value, record.output_units,
            record.additivity_gap, round(record.shap_ms, 2), record.input_hash,
        ]
        answers = [m for m in record.messages if m.role == "assistant"]
        if not answers:
            writer.writerow(base + [""] * 14)
            continue
        # Each audience is its own thread, so questions are paired per audience
        last_question: Dict[str, str] = {}
        for m in record.messages:
            audience = m.audience or DEFAULT_AUDIENCE.value
            if m.role == "user":
                last_question[audience] = m.content
                continue
            f = m.faithfulness or {}
            writer.writerow(base + [
                m.kind, audience, last_question.get(audience, "") if m.kind == "followup" else "", m.llm_model,
                round(m.latency_ms or 0, 2), round(m.ttft_ms or 0, 2) if m.ttft_ms else "",
                f.get("numbers_checked"), f.get("numbers_verified"), f.get("fabricated"),
                f.get("structure_complete", ""), f.get("direction_errors", ""),
                json.dumps([x["feature"] for x in f.get("misplaced_features", [])]),
                json.dumps(f.get("features_mentioned", [])),
                m.content,
            ])

    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="model_{model_id}_audit.csv"'},
    )
