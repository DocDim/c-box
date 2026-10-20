from typing import List, Dict, Any
from pydantic import BaseModel

class DatasetPreviewResponse(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns: List[str]
    preview_rows: List[Dict[str, Any]]