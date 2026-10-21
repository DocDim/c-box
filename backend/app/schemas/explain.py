from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel

class FeatureContribution(BaseModel):
    feature: str
    actual_value: Union[float, int, str]
    shap_value: float

class InstanceExplainRequest(BaseModel):
    model_id: int
    instance_features: Dict[str, Any]

class InstanceExplainResponse(BaseModel):
    model_id: int
    prediction: float
    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    base_value: float
    feature_contributions: List[FeatureContribution]

class GlobalImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float

class GlobalExplainResponse(BaseModel):
    model_id: int
    samples_evaluated: int
    global_importance: List[GlobalImportanceItem]