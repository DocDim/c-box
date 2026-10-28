import hashlib
import json
import time
from typing import AsyncIterator, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.models.user import User
from app.models.artifact import AccessRole, ModelArtifact
from app.models.explanation import ChatMessage, ExplanationRecord
from app.schemas.chat import ChatMessageResponse, ChatRequest, ChatResponse, ConversationResponse
from app.schemas.explain import GlobalExplainResponse, InstanceExplainRequest, InstanceExplainResponse
from app.core.dependencies import get_current_user
from app.core.faithfulness import check_faithfulness
from app.core.feature_alignment import align_frame, read_dataset_frame
from app.core.instances import get_engine, resolve_instance
from app.core.llm_service import (
    ChatTurn,
    GenerationStats,
    LLMUnavailableError,
    build_context_payload,
    llm_service,
)
from app.core.prompt_builder import DEFAULT_AUDIENCE, Audience, ModelMetadata, PromptBuilder
from app.core.permissions import verify_dataset_access, verify_model_access

router = APIRouter(prefix="/explain", tags=["Explainability Engine"])


def _record_response(record: ExplanationRecord) -> InstanceExplainResponse:
    return InstanceExplainResponse(
        explanation_id=record.id,
        model_id=record.model_id,
        task_type=record.task_type,
        explainer_type=record.explainer_type,
        prediction=record.prediction,
        predicted_class=record.predicted_class,
        explained_class=record.explained_class,
        probabilities=record.probabilities,
        base_value=record.base_value,
        explained_output=record.explained_output,
        output_units=record.output_units,
        additivity_gap=record.additivity_gap,
        feature_contributions=record.feature_contributions,
        input_features=record.input_features,
        dataset_id=record.dataset_id,
        row_index=record.row_index,
        shap_ms=record.shap_ms,
        created_at=record.created_at,
    )


@router.post("/instance", response_model=InstanceExplainResponse)
async def explain_instance(
    payload: InstanceExplainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Computes local SHAP attributions (FR-3.3) and stores them server-side. The returned
    explanation_id is the only handle the LLM chat accepts, so narratives are always
    grounded on values the server computed.
    """
    model_record = await verify_model_access(
        payload.model_id, current_user, db, min_role=AccessRole.EXPLAINER
    )
    instance_df, input_features = await resolve_instance(payload, model_record, current_user, db)
    engine = await get_engine(model_record, db)

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(
            engine.explain_instance, instance_df, settings.SHAP_ADDITIVITY_TOLERANCE
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation failed: {e}"
        )
    shap_ms = (time.perf_counter() - started) * 1000

    record = ExplanationRecord(
        model_id=model_record.id,
        user_id=current_user.id,
        dataset_id=payload.dataset_id if payload.row_index is not None else None,
        row_index=payload.row_index if payload.dataset_id is not None else None,
        input_hash=hashlib.sha256(
            json.dumps(input_features, sort_keys=True, default=str).encode()
        ).hexdigest(),
        input_features=input_features,
        shap_ms=shap_ms,
        **{k: result[k] for k in (
            "task_type", "explainer_type", "predicted_class", "explained_class", "probabilities",
            "prediction", "base_value", "explained_output", "output_units", "additivity_gap",
            "feature_contributions",
        )},
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return _record_response(record)


@router.get("/global/{model_id}", response_model=GlobalExplainResponse)
async def explain_global(
    model_id: int,
    dataset_id: Optional[int] = Query(None, description="Defaults to the model's background data"),
    max_samples: int = Query(200, ge=10, le=300),
    class_index: Optional[int] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Global importance plus beeswarm / dependence data over a sample of N <= 300 rows (FR-3.5),
    computed on the model's background data unless another dataset is named.
    """
    model_record = await verify_model_access(model_id, current_user, db, min_role=AccessRole.EXPLAINER)
    if dataset_id is None:
        if model_record.background_dataset_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="This model has no background data yet. The model owner can upload it on the Models page."
            )
        dataset_id = model_record.background_dataset_id
    dataset = await verify_dataset_access(dataset_id, current_user, db)
    engine = await get_engine(model_record, db)

    def _run():
        frame = align_frame(read_dataset_frame(dataset), model_record, dataset.target_column)
        return engine.explain_global(frame, max_samples=max_samples, class_index=class_index)

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global SHAP explanation failed: {e}"
        )
    return GlobalExplainResponse(
        model_id=model_record.id,
        dataset_id=dataset.id,
        shap_ms=(time.perf_counter() - started) * 1000,
        **result,
    )


async def _load_owned_explanation(
    explanation_id: str, user: User, db: AsyncSession
) -> Tuple[ExplanationRecord, ModelArtifact]:
    record = await db.get(ExplanationRecord, explanation_id)
    # A conversation belongs to the user who ran the explanation; 404 avoids leaking ids
    if record is None or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Explanation not found.")
    model = await verify_model_access(record.model_id, user, db, min_role=AccessRole.EXPLAINER)
    return record, model


async def _load_history(explanation_id: str, audience: Audience, db: AsyncSession) -> List[ChatMessage]:
    """The conversation thread for one audience (rows without one predate audiences)."""
    messages = (await db.execute(
        select(ChatMessage).where(ChatMessage.explanation_id == explanation_id).order_by(ChatMessage.id)
    )).scalars().all()
    return [m for m in messages if (m.audience or DEFAULT_AUDIENCE.value) == audience.value]


async def _persist_turn(
    explanation_id: str,
    audience: Audience,
    question: Optional[str],
    stats: GenerationStats,
    faithfulness: dict,
) -> ChatMessage:
    """Saves the question (if any) and the assistant answer after a successful generation."""
    async with AsyncSessionLocal() as session:
        if question is not None:
            session.add(ChatMessage(
                explanation_id=explanation_id, role="user", kind="followup",
                audience=audience.value, content=question,
            ))
        assistant = ChatMessage(
            explanation_id=explanation_id,
            role="assistant",
            kind="followup" if question is not None else "narrative",
            audience=audience.value,
            content=stats.text,
            llm_model=stats.model,
            latency_ms=stats.latency_ms,
            ttft_ms=stats.ttft_ms,
            faithfulness=faithfulness,
        )
        session.add(assistant)
        await session.commit()
        await session.refresh(assistant)
        return assistant


async def _prepare_chat(payload: ChatRequest, user: User, db: AsyncSession):
    record, model = await _load_owned_explanation(payload.explanation_id, user, db)
    history = await _load_history(record.id, payload.audience, db)
    existing_narrative = next(
        (m for m in history if m.role == "assistant" and m.kind == "narrative"), None
    )
    prompt = PromptBuilder(ModelMetadata.from_artifact(model), payload.audience)
    context = build_context_payload(record, prompt, settings.LLM_TOP_K_FEATURES)
    messages = llm_service.build_messages(
        prompt,
        context,
        [ChatTurn(role=m.role, kind=m.kind, content=m.content) for m in history],
        payload.question,
    )
    return record, prompt, context, messages, existing_narrative


def _check(text: str, context: dict, prompt: PromptBuilder, question: Optional[str]) -> dict:
    return check_faithfulness(
        text, context, expect_sections=question is None, section_headings=prompt.section_headings
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Grounded narrative (no question) or multi-turn follow-up (FR-4.1 - FR-4.4).
    Every answer is scored by the automated faithfulness checker.
    """
    record, prompt, context, messages, existing = await _prepare_chat(payload, current_user, db)
    if payload.question is None and existing is not None:
        return ChatResponse(explanation_id=record.id, message=ChatMessageResponse.model_validate(existing))

    try:
        stats = await llm_service.generate(messages)
    except LLMUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    faithfulness = _check(stats.text, context, prompt, payload.question)
    message = await _persist_turn(record.id, payload.audience, payload.question, stats, faithfulness)
    return ChatResponse(explanation_id=record.id, message=ChatMessageResponse.model_validate(message))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Server-Sent Events variant of /chat: streams tokens as Gemma produces them (NFR-3)."""
    record, prompt, context, messages, existing = await _prepare_chat(payload, current_user, db)

    async def events() -> AsyncIterator[str]:
        if payload.question is None and existing is not None:
            yield _sse("done", ChatMessageResponse.model_validate(existing).model_dump(mode="json"))
            return
        stats = GenerationStats()
        try:
            async for token in llm_service.stream_chat(messages, stats):
                yield _sse("token", {"text": token})
        except LLMUnavailableError as e:
            yield _sse("error", {"detail": str(e)})
            return
        faithfulness = _check(stats.text, context, prompt, payload.question)
        message = await _persist_turn(record.id, payload.audience, payload.question, stats, faithfulness)
        yield _sse("done", ChatMessageResponse.model_validate(message).model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{explanation_id}", response_model=InstanceExplainResponse)
async def get_explanation(
    explanation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    record, _ = await _load_owned_explanation(explanation_id, current_user, db)
    return _record_response(record)


@router.get("/{explanation_id}/conversation", response_model=ConversationResponse)
async def get_conversation(
    explanation_id: str,
    audience: Audience = Query(DEFAULT_AUDIENCE),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    record, _ = await _load_owned_explanation(explanation_id, current_user, db)
    history = await _load_history(record.id, audience, db)
    return ConversationResponse(
        explanation_id=record.id,
        messages=[ChatMessageResponse.model_validate(m) for m in history],
    )
