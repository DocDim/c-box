import math
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple

# Output-unit labels reported alongside every explanation so that baseline,
# attributions and prediction are always interpreted on the same scale.
UNITS_PROBABILITY = "probability"
UNITS_LOG_ODDS = "log-odds"
UNITS_TARGET = "target units"
UNITS_MARGIN = "raw model margin"


class BackgroundDataRequired(ValueError):
    """Raised when a non-tree model is explained without reference data."""


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def normalize_shap_output(raw: Any) -> np.ndarray:
    """
    Normalizes the many SHAP return shapes (list per class, 2D, 3D) into a
    (n_rows, n_features, n_outputs) array. n_outputs is 1 for single-output models.
    """
    if isinstance(raw, list):
        arr = np.stack([np.asarray(r, dtype=float) for r in raw], axis=-1)
    else:
        arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def normalize_expected_value(expected: Any, n_outputs: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    if arr.size == 1 and n_outputs > 1:
        arr = np.repeat(arr, n_outputs)
    return arr


class ShapExplainerEngine:
    def __init__(
        self,
        model: Any,
        task_type: str,
        background_data: Optional[pd.DataFrame] = None,
        background_max_rows: int = 100,
        kmeans_k: int = 30,
    ):
        """
        Initializes SHAP explainer with automated algorithm selection (FR-3.2).
        Tree ensembles are routed to TreeExplainer; other estimators fall back to
        KernelExplainer over a k-means summary of the reference dataset.
        """
        self.model = model
        self.task_type = task_type
        self.is_classifier = task_type == "classification" and hasattr(model, "predict_proba")
        self.classes: List[str] = [str(c) for c in getattr(model, "classes_", [])]
        self.background = self._sample_background(background_data, background_max_rows)
        self.kmeans_k = kmeans_k
        self.explainer, self.explainer_type = self._init_explainer()

    @staticmethod
    def _sample_background(df: Optional[pd.DataFrame], max_rows: int) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return None
        if len(df) > max_rows:
            return df.sample(n=max_rows, random_state=0)
        return df

    def _predict_fn(self, X):
        return self.model.predict_proba(X) if self.is_classifier else self.model.predict(X)

    def _init_explainer(self) -> Tuple[Any, str]:
        # Fast tree path. With reference data, classifiers are explained directly in
        # probability space (interventional), so baseline + sum(phi) == P(class).
        if self.background is not None and self.is_classifier:
            try:
                explainer = shap.TreeExplainer(
                    self.model,
                    data=self.background,
                    model_output="probability",
                    feature_perturbation="interventional",
                )
                return explainer, "TreeExplainer (interventional, probability)"
            except Exception:
                pass
        try:
            return shap.TreeExplainer(self.model), "TreeExplainer (path-dependent)"
        except Exception:
            pass

        # Kernel fallback for black-box estimators (needs a background distribution)
        if self.background is None:
            raise BackgroundDataRequired(
                f"'{type(self.model).__name__}' is not a tree ensemble, so SHAP needs reference data. "
                "Upload a dataset linked to this model to use as the background distribution."
            )
        k = min(self.kmeans_k, len(self.background))
        summary = shap.kmeans(self.background, k) if len(self.background) > k else self.background
        return shap.KernelExplainer(self._predict_fn, summary), "KernelExplainer (k-means background)"

    def _shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.explainer, shap.KernelExplainer):
            raw = self.explainer.shap_values(X, silent=True)
        else:
            raw = self.explainer.shap_values(X, check_additivity=False)
        values = normalize_shap_output(raw)
        expected = normalize_expected_value(self.explainer.expected_value, values.shape[2])
        return values, expected

    def _output_index_for_class(self, class_idx: int, n_outputs: int) -> Tuple[int, int]:
        """
        Maps a class index onto the SHAP output column. Single-output classifiers
        (e.g. binary GradientBoosting) only explain the positive class.
        Returns (output_column, explained_class_index).
        """
        if n_outputs == 1:
            return 0, (1 if len(self.classes) == 2 else class_idx)
        return min(class_idx, n_outputs - 1), min(class_idx, n_outputs - 1)

    @staticmethod
    def _detect_units(explained: float, candidates: Dict[str, float], tolerance: float) -> Tuple[str, float]:
        """Picks the output scale whose reference value matches base + sum(phi)."""
        best_units, best_gap = UNITS_MARGIN, float("inf")
        for units, reference in candidates.items():
            gap = abs(explained - reference)
            if gap < best_gap:
                best_units, best_gap = units, gap
        if best_gap > max(tolerance, 1e-2 * max(1.0, abs(explained))):
            return UNITS_MARGIN, best_gap
        return best_units, best_gap

    def explain_instance(self, instance_df: pd.DataFrame, additivity_tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Computes local attributions for one row (FR-3.3) and reports the prediction on the
        same scale as the attributions, plus the additivity gap |base + sum(phi) - f(x)|.
        """
        feature_names = list(instance_df.columns)
        values, expected = self._shap_values(instance_df)
        n_outputs = values.shape[2]

        predicted_class: Optional[str] = None
        explained_class: Optional[str] = None
        probabilities: Optional[Dict[str, float]] = None

        if self.is_classifier:
            probs = np.asarray(self.model.predict_proba(instance_df)[0], dtype=float)
            labels = self.classes or [str(i) for i in range(len(probs))]
            pred_idx = int(np.argmax(probs))
            predicted_class = labels[pred_idx]
            probabilities = {lbl: float(p) for lbl, p in zip(labels, probs)}

            out_col, explained_idx = self._output_index_for_class(pred_idx, n_outputs)
            explained_class = labels[explained_idx]
            prediction = float(probs[explained_idx])
            candidates = {UNITS_PROBABILITY: prediction, UNITS_LOG_ODDS: _logit(prediction)}
        else:
            out_col = 0
            prediction = float(np.asarray(self.model.predict(instance_df)).ravel()[0])
            candidates = {UNITS_TARGET: prediction}

        shap_vec = values[0, :, out_col]
        base_value = float(expected[out_col])
        explained_output = base_value + float(shap_vec.sum())
        output_units, additivity_gap = self._detect_units(explained_output, candidates, additivity_tolerance)

        # Single-output binary models explain the positive class. When the negative class
        # wins, mirror the attributions (p0 = 1 - p1, logit(p0) = -logit(p1)) so the
        # explanation always concerns the predicted class.
        if (
            self.is_classifier and n_outputs == 1 and probabilities is not None
            and len(probabilities) == 2 and pred_idx == 0
            and output_units in (UNITS_PROBABILITY, UNITS_LOG_ODDS)
        ):
            shap_vec = -shap_vec
            base_value = (1.0 - base_value) if output_units == UNITS_PROBABILITY else -base_value
            explained_class = predicted_class
            prediction = float(probs[0])
            explained_output = base_value + float(shap_vec.sum())
            reference = prediction if output_units == UNITS_PROBABILITY else _logit(prediction)
            additivity_gap = abs(explained_output - reference)

        contributions: List[Dict[str, Any]] = []
        for name, s_val in zip(feature_names, shap_vec):
            raw_val = instance_df[name].values[0]
            contributions.append({
                "feature": str(name),
                "actual_value": (
                    float(raw_val)
                    if isinstance(raw_val, (np.number, int, float)) and not isinstance(raw_val, bool)
                    else str(raw_val)
                ),
                "shap_value": float(s_val)
            })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "task_type": self.task_type,
            "explainer_type": self.explainer_type,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "explained_class": explained_class,
            "probabilities": probabilities,
            "base_value": base_value,
            "explained_output": explained_output,
            "output_units": output_units,
            "additivity_gap": float(additivity_gap),
            "feature_contributions": contributions,
        }

    def explain_global(
        self,
        dataset_df: pd.DataFrame,
        max_samples: int = 300,
        class_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Global SHAP distribution over a representative sample (FR-3.5). Returns mean |phi|
        for the bar chart plus the per-sample matrix for beeswarm / dependence plots.
        """
        if isinstance(self.explainer, shap.KernelExplainer):
            # KernelExplainer costs O(rows x coalitions x background); keep global runs tractable
            max_samples = min(max_samples, 50)
        eval_df = dataset_df.sample(n=max_samples, random_state=0) if len(dataset_df) > max_samples else dataset_df
        values, expected = self._shap_values(eval_df)
        n_outputs = values.shape[2]
        labels = self.classes or [str(i) for i in range(n_outputs)]

        if n_outputs == 1:
            matrix = values[:, :, 0]
            base_value = float(expected[0])
            if self.is_classifier and len(labels) == 2:
                mode = f"class {labels[1]}"
            elif self.is_classifier:
                mode = f"class {labels[0]}"
            else:
                mode = "regression output"
            importance = np.mean(np.abs(matrix), axis=0)
        elif class_index is not None:
            col = min(max(class_index, 0), n_outputs - 1)
            matrix = values[:, :, col]
            base_value = float(expected[col])
            mode = f"class {labels[col]}"
            importance = np.mean(np.abs(matrix), axis=0)
        elif n_outputs == 2:
            matrix = values[:, :, 1]
            base_value = float(expected[1])
            mode = f"class {labels[1]}"
            importance = np.mean(np.abs(matrix), axis=0)
        else:
            # Multiclass: importance averaged over every class; beeswarm shows each
            # sample's attributions toward its own predicted class.
            importance = np.mean(np.abs(values), axis=(0, 2))
            pred_idx = np.argmax(np.asarray(self.model.predict_proba(eval_df)), axis=1)
            pred_idx = np.minimum(pred_idx, n_outputs - 1)
            matrix = values[np.arange(len(eval_df)), :, pred_idx]
            base_value = float(np.mean(expected))
            mode = "each sample's predicted class (importance averaged over all classes)"

        order = np.argsort(-importance)
        columns = list(eval_df.columns)
        global_importance = [
            {"feature": str(columns[i]), "mean_abs_shap": float(importance[i])} for i in order
        ]

        feature_values: List[List[Optional[float]]] = []
        numeric = eval_df.apply(pd.to_numeric, errors="coerce")
        for i in order:
            col = numeric.iloc[:, i]
            feature_values.append([None if pd.isna(v) else float(v) for v in col])

        return {
            "samples_evaluated": int(len(eval_df)),
            "explained_output": mode,
            "base_value": base_value,
            "explainer_type": self.explainer_type,
            "global_importance": global_importance,
            "beeswarm": {
                "features": [str(columns[i]) for i in order],
                "shap_values": [[float(v) for v in matrix[:, i]] for i in order],
                "feature_values": feature_values,
            },
        }
