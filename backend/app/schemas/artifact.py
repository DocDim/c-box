from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, EmailStr, ConfigDict, Field
from app.models.artifact import AccessRole

class ShareGrantRequest(BaseModel):
    user_email: EmailStr
    role: AccessRole = AccessRole.EXPLAINER

class SharedUserResponse(BaseModel):
    user_id: int
    email: str
    role: AccessRole
    granted_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ModelResponse(BaseModel):
    id: int
    name: str
    domain_context: str
    estimator_type: str
    owner_id: int
    owner_email: Optional[str] = None
    is_owner: bool
    assigned_role: str
    created_at: datetime
    task_type: str = "unknown"
    file_format: str = "joblib"
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    class_labels: Optional[List[str]] = None
    background_dataset_id: Optional[int] = None
    feature_glossary: Optional[Dict[str, str]] = None
    outcome_description: Optional[str] = None
    class_descriptions: Optional[Dict[str, str]] = None
    decision_context: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class ModelMetadataUpdate(BaseModel):
    """Owner-edited business context used by the LLM prompt layer. Blank values clear a field."""
    feature_glossary: Dict[str, str] = Field(default_factory=dict)
    outcome_description: Optional[str] = Field(default=None, max_length=1000)
    class_descriptions: Dict[str, str] = Field(default_factory=dict)
    decision_context: Optional[str] = Field(default=None, max_length=2000)
