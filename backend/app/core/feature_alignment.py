from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from app.models.artifact import ModelArtifact
from app.models.dataset import Dataset
from app.core.storage import ensure_within_storage


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns that are fully numeric (e.g. '5.1' from a form) to numbers."""
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted
    return df


def expected_feature_names(model: ModelArtifact, fallback_columns: List[str]) -> List[str]:
    """Feature order the estimator was fitted on; falls back to the supplied column order."""
    if model.feature_names:
        return list(model.feature_names)
    if model.n_features is not None and len(fallback_columns) != model.n_features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dimension mismatch: model expects {model.n_features} features, received {len(fallback_columns)}."
        )
    return list(fallback_columns)


def align_frame(df: pd.DataFrame, model: ModelArtifact, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Drops the target column, validates that every model feature is present and
    reorders columns to the fitted feature order (FR-2.5).
    """
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    features = expected_feature_names(model, list(df.columns))
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing model features: {', '.join(missing[:10])}"
        )
    return _coerce_numeric_columns(df[features].copy())


def align_instance(features: Dict[str, Any], model: ModelArtifact) -> pd.DataFrame:
    """Builds a single-row frame for manual / selected-row inputs."""
    return align_frame(pd.DataFrame([features]), model)


def read_dataset_frame(dataset: Dataset) -> pd.DataFrame:
    path = ensure_within_storage(dataset.file_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file missing on disk.")
    return pd.read_csv(path)


def validate_dataset_columns(
    columns: List[str], model: Optional[ModelArtifact], target_column: Optional[str]
) -> Dict[str, Any]:
    """
    Checks CSV columns against the model's fitted features. If exactly one extra
    column exists and no target was named, it is treated as the target.
    """
    if target_column and target_column not in columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{target_column}' not found in CSV."
        )
    if model is None:
        return {"target_column": target_column, "missing": [], "extra": []}

    candidate_cols = [c for c in columns if c != target_column]
    if model.feature_names:
        missing = [f for f in model.feature_names if f not in candidate_cols]
        extra = [c for c in candidate_cols if c not in model.feature_names]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"CSV is missing model features: {', '.join(missing[:10])}"
            )
        if not target_column and len(extra) == 1:
            target_column, extra = extra[0], []
        return {"target_column": target_column, "missing": [], "extra": extra}

    if model.n_features is not None and len(candidate_cols) != model.n_features:
        hint = " Specify the target column." if len(candidate_cols) == model.n_features + 1 else ""
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model expects {model.n_features} features but CSV has {len(candidate_cols)} columns.{hint}"
        )
    return {"target_column": target_column, "missing": [], "extra": []}


def summarize_frame(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Per-column summary statistics for the dataset preview (AC-2)."""
    summary: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        entry: Dict[str, Any] = {"dtype": str(series.dtype), "missing": int(series.isna().sum())}
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            for key in ("mean", "std", "min", "max"):
                val = desc.get(key)
                entry[key] = None if val is None or pd.isna(val) else float(val)
        else:
            entry["unique"] = int(series.nunique())
        summary[str(col)] = entry
    return summary


def frame_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-safe records (NaN -> None, numpy scalars -> Python)."""
    clean = df.astype(object).where(pd.notna(df), None)
    records = clean.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, np.generic):
                row[k] = v.item()
    return records
