from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.artifact import AccessRole
from app.schemas.chat import ChatExplanationRequest, ChatExplanationResponse
from app.core.dependencies import get_current_user
from app.core.permissions import verify_model_access
from app.core.llm_service import llm_service

router = APIRouter(prefix="/chat", tags=["LLM Interpretation & Chat"])

@router.post("/chat", response_model=ChatExplanationResponse)
async def generate_narrative_explanation(
    payload: ChatExplanationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Translates SHAP metrics into grounded natural language narratives via Gemma/Ollama,
    supporting contextual follow-up questions.
    """
    model_record = await verify_model_access(
        payload.model_id, current_user, db, min_role=AccessRole.EXPLAINER
    )

    feature_dicts = [f.model_dump() for f in payload.feature_contributions]

    serialized_context = llm_service.serialize_shap_context(
        domain_context=model_record.domain_context,
        prediction=payload.prediction,
        base_value=payload.base_value,
        feature_contributions=feature_dicts
    )

    narrative = await llm_service.generate_explanation(
        domain_context=model_record.domain_context,
        serialized_context=serialized_context,
        user_query=payload.user_query
    )

    return ChatExplanationResponse(explanation=narrative)