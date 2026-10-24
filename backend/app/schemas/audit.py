from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.schemas.chat import ChatMessageResponse
from app.schemas.explain import FeatureContribution

class AuditExplanation(BaseModel):
    explanation_id: str
    user_email: str
    created_at: datetime
    explainer_type: str
    task_type: str
    predicted_class: Optional[str] = None
    prediction: float
    base_value: float
    output_units: str
    additivity_gap: float
    shap_ms: float
    input_hash: str
    input_features: Dict[str, Any]
    dataset_id: Optional[int] = None
    row_index: Optional[int] = None
    feature_contributions: List[FeatureContribution]
    messages: List[ChatMessageResponse]

class AuditListResponse(BaseModel):
    model_id: int
    total: int
    items: List[AuditExplanation]

class LatencyStats(BaseModel):
    count: int
    mean: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None

class AuditMetricsResponse(BaseModel):
    model_id: int
    explanations: int
    llm_responses: int
    narratives: int
    followups: int
    shap_ms: LatencyStats
    llm_latency_ms: LatencyStats
    llm_ttft_ms: LatencyStats
    max_additivity_gap: Optional[float] = None
    faithfulness_zero_fabrication_rate: Optional[float] = None
    faithfulness_numbers_verified_rate: Optional[float] = None
    narrative_structure_complete_rate: Optional[float] = None
    narrative_direction_error_rate: Optional[float] = None
    mean_features_mentioned: Optional[float] = None
