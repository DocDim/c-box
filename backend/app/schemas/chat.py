from typing import List, Optional
from pydantic import BaseModel
from app.schemas.explain import FeatureContribution

class ChatExplanationRequest(BaseModel):
    model_id: int
    prediction: float
    base_value: float
    feature_contributions: List[FeatureContribution]
    user_query: Optional[str] = None

class ChatExplanationResponse(BaseModel):
    explanation: str