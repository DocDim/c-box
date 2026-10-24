from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field, model_validator

class FeatureContribution(BaseModel):
    feature: str
    actual_value: Union[float, int, str, None]
    shap_value: float

class InstanceSource(BaseModel):
    """Either manual feature values, or a row of a stored dataset (FR-3.1)."""
    model_id: int
    instance_features: Optional[Dict[str, Any]] = None
    dataset_id: Optional[int] = None
    row_index: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_source(self) -> "InstanceSource":
        has_row = self.dataset_id is not None and self.row_index is not None
        if not self.instance_features and not has_row:
            raise ValueError("Provide instance_features, or dataset_id together with row_index.")
        return self

class InstanceExplainRequest(InstanceSource):
    pass

class PredictRequest(InstanceSource):
    pass

class PredictResponse(BaseModel):
    model_id: int
    task_type: str
    prediction: float
    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    input_features: Dict[str, Any]

class InstanceExplainResponse(BaseModel):
    explanation_id: str
    model_id: int
    task_type: str
    explainer_type: str
    prediction: float
    predicted_class: Optional[str] = None
    explained_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    base_value: float
    explained_output: float
    output_units: str
    additivity_gap: float
    feature_contributions: List[FeatureContribution]
    input_features: Dict[str, Any]
    dataset_id: Optional[int] = None
    row_index: Optional[int] = None
    shap_ms: float
    created_at: datetime

class GlobalImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float

class BeeswarmData(BaseModel):
    features: List[str]
    shap_values: List[List[float]]
    feature_values: List[List[Optional[float]]]

class GlobalExplainResponse(BaseModel):
    model_id: int
    dataset_id: int
    samples_evaluated: int
    explained_output: str
    explainer_type: str
    base_value: float
    global_importance: List[GlobalImportanceItem]
    beeswarm: BeeswarmData
    shap_ms: float

class BatchPredictRequest(BaseModel):
    model_id: int
    dataset_id: int
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

class BatchPredictionRow(BaseModel):
    row_index: int
    features: Dict[str, Any]
    prediction: Optional[float] = None
    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    actual: Optional[Any] = None

class BatchPredictResponse(BaseModel):
    model_id: int
    dataset_id: int
    task_type: str
    total_rows: int
    offset: int
    limit: int
    target_column: Optional[str] = None
    class_counts: Optional[Dict[str, int]] = None
    prediction_stats: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None
    predict_ms: float
    rows: List[BatchPredictionRow]
