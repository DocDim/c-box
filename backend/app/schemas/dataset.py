from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DatasetResponse(BaseModel):
    id: int
    name: str
    owner_id: int
    model_id: Optional[int] = None
    n_rows: int
    columns: List[str]
    target_column: Optional[str] = None
    created_at: datetime
    purpose: str = "evaluation"
    is_background: bool = False

class DatasetPreviewResponse(DatasetResponse):
    extra_columns: List[str] = []
    summary: Dict[str, Dict[str, Any]] = {}
    preview_rows: List[Dict[str, Any]]

class DatasetRowsResponse(BaseModel):
    dataset_id: int
    offset: int
    limit: int
    total_rows: int
    columns: List[str]
    rows: List[Dict[str, Any]]
