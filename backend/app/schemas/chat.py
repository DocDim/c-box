from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class ChatRequest(BaseModel):
    """
    The LLM is grounded only on the server-stored explanation referenced here;
    attribution values are never accepted from the client.
    Omit `question` to generate the 4-part narrative.
    """
    explanation_id: str = Field(min_length=1, max_length=32)
    question: Optional[str] = Field(default=None, min_length=1, max_length=1000)

class ChatMessageResponse(BaseModel):
    id: int
    role: str
    kind: str
    content: str
    llm_model: Optional[str] = None
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    faithfulness: Optional[Dict[str, Any]] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ChatResponse(BaseModel):
    explanation_id: str
    message: ChatMessageResponse

class ConversationResponse(BaseModel):
    explanation_id: str
    messages: List[ChatMessageResponse]
