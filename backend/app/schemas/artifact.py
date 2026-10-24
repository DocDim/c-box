from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, ConfigDict
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
    model_config = ConfigDict(from_attributes=True)
