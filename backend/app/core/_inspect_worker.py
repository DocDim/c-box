"""
Runs inside a short-lived child process. Deliberately imports nothing from `app`
so that deserializing an untrusted artifact cannot touch the server's settings,
database session, or open sockets, and a crash or hang only kills the child.
"""
from typing import Any, Dict, List, Optional

# Module prefixes whose types are allowed when loading .skops artifacts
SKOPS_TRUSTED_PREFIXES = (
    "sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.", "builtins.", "collections.",
)


def load_artifact(path: str, file_format: str) -> Any:
    if file_format == "skops":
        import skops.io as sio

        untrusted = sio.get_untrusted_types(file=path)
        rejected = [t for t in untrusted if not t.startswith(SKOPS_TRUSTED_PREFIXES)]
        if rejected:
            raise ValueError(f"Artifact references untrusted types: {', '.join(rejected[:5])}")
        return sio.load(path, trusted=untrusted)

    import joblib
    return joblib.load(path)


def _feature_names(model: Any) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return [str(f) for f in model.feature_names_in_]
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        names = booster().feature_names
        if names:
            return [str(f) for f in names]
    if hasattr(model, "feature_name_"):
        return [str(f) for f in model.feature_name_]
    return None


def inspect_model(model: Any) -> Dict[str, Any]:
    """Extracts estimator architecture, task type and expected input features (FR-2.2)."""
    if not callable(getattr(model, "predict", None)):
        raise ValueError(f"Object of type '{type(model).__name__}' has no predict() method.")

    task_type = "unknown"
    try:
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(model):
            task_type = "classification"
        elif is_regressor(model):
            task_type = "regression"
    except Exception:
        pass
    if task_type == "unknown":
        task_type = "classification" if hasattr(model, "predict_proba") else "regression"

    feature_names = _feature_names(model)
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None and feature_names is not None:
        n_features = len(feature_names)

    classes = getattr(model, "classes_", None)
    class_labels = [str(c) for c in classes] if classes is not None else None

    return {
        "estimator_type": type(model).__name__,
        "task_type": task_type,
        "n_features": int(n_features) if n_features is not None else None,
        "feature_names": feature_names,
        "class_labels": class_labels,
    }


def inspect_artifact(path: str, file_format: str) -> Dict[str, Any]:
    """Entry point executed in the child process."""
    return inspect_model(load_artifact(path, file_format))
