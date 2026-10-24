from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def predict_frame(estimator: Any, task_type: str, instance_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Single source of truth for point predictions, shared by /predict and the SHAP engine
    so both report the same class labels and the same score (P(predicted class)).
    """
    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None

    if task_type == "classification" and hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(instance_df)[0], dtype=float)
        labels = [str(c) for c in getattr(estimator, "classes_", range(len(probs)))]
        idx = int(np.argmax(probs))
        predicted_class = labels[idx]
        probabilities = {lbl: float(p) for lbl, p in zip(labels, probs)}
        prediction = float(probs[idx])
    else:
        raw = np.asarray(estimator.predict(instance_df)).ravel()[0]
        try:
            prediction = float(raw)
        except (TypeError, ValueError):
            prediction = float("nan")
            predicted_class = str(raw)

    return {
        "task_type": task_type,
        "prediction": prediction,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
    }


def _label(value: Any) -> Optional[str]:
    """Normalizes target values so 1, 1.0 and '1' compare equal to class label '1'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def predict_batch(
    estimator: Any, task_type: str, frame: pd.DataFrame, actual: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Vectorized predictions for a whole dataset plus summary statistics and, when the
    dataset carries a target column, accuracy (classification) or MAE / RMSE / R² (regression).
    """
    result: Dict[str, Any] = {"task_type": task_type, "class_counts": None, "prediction_stats": None, "metrics": None}

    if task_type == "classification" and hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(frame), dtype=float)
        labels = [str(c) for c in getattr(estimator, "classes_", range(probs.shape[1]))]
        idx = probs.argmax(axis=1)
        predicted = [labels[i] for i in idx]
        result.update(
            labels=labels,
            predicted_class=predicted,
            prediction=probs[np.arange(len(idx)), idx].tolist(),
            probabilities=[{lbl: float(p) for lbl, p in zip(labels, row)} for row in probs],
            class_counts={lbl: int(sum(1 for p in predicted if p == lbl)) for lbl in labels},
        )
        if actual is not None:
            truth = [_label(v) for v in actual]
            pairs = [(t, p) for t, p in zip(truth, predicted) if t is not None]
            if pairs:
                result["metrics"] = {
                    "accuracy": float(np.mean([t == p for t, p in pairs])),
                    "evaluated_rows": float(len(pairs)),
                }
    else:
        values = np.asarray(estimator.predict(frame)).ravel()
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            predicted = [str(v) for v in values]
            result.update(predicted_class=predicted, prediction=[None] * len(values), probabilities=None,
                          class_counts={v: predicted.count(v) for v in sorted(set(predicted))})
        else:
            result.update(predicted_class=None, prediction=numeric.tolist(), probabilities=None,
                          prediction_stats={"mean": float(numeric.mean()), "std": float(numeric.std()),
                                            "min": float(numeric.min()), "max": float(numeric.max())})
            if actual is not None:
                truth = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
                mask = ~np.isnan(truth)
                if mask.any():
                    err = numeric[mask] - truth[mask]
                    ss_tot = float(((truth[mask] - truth[mask].mean()) ** 2).sum())
                    metrics = {
                        "mae": float(np.abs(err).mean()),
                        "rmse": float(np.sqrt((err ** 2).mean())),
                        "evaluated_rows": float(mask.sum()),
                    }
                    if ss_tot > 0:  # R² is undefined for a constant target
                        metrics["r2"] = float(1 - (err ** 2).sum() / ss_tot)
                    result["metrics"] = metrics
    return result
