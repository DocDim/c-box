import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional

class ShapExplainerEngine:
    def __init__(self, model: Any, background_data: Optional[pd.DataFrame] = None):
        """
        Initializes SHAP explainer with automated algorithm selection.
        Routes tree ensembles to TreeExplainer and black-box models to KernelExplainer.
        """
        self.model = model
        self.background_data = background_data
        self.explainer = self._init_explainer()

    def _init_explainer(self):
        try:
            return shap.TreeExplainer(self.model)
        except Exception:
            if self.background_data is None or len(self.background_data) == 0:
                raise ValueError("KernelExplainer requires non-empty background evaluation data.")
            
            k_samples = min(30, len(self.background_data))
            bg_summary = shap.kmeans(self.background_data, k_samples)
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            return shap.KernelExplainer(predict_fn, bg_summary)

    def explain_instance(self, instance_df: pd.DataFrame, target_class_index: int = 1) -> Dict[str, Any]:
        """
        Calculates local feature attribution for a single row, extracting
        predicted class labels, full probability distribution, and SHAP vectors.
        """
        feature_names = list(instance_df.columns)
        raw_shap_values = self.explainer.shap_values(instance_df)

        predicted_class: Optional[str] = None
        probabilities: Optional[Dict[str, float]] = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(instance_df)[0]
            pred_class_idx = int(np.argmax(probs))
            
            classes = getattr(self.model, "classes_", list(range(len(probs))))
            predicted_class = str(classes[pred_class_idx])
            prediction_score = float(probs[pred_class_idx])
            probabilities = {str(c): float(p) for c, p in zip(classes, probs)}
            selected_class_idx = pred_class_idx
        else:
            pred_raw = self.model.predict(instance_df)[0]
            predicted_class = str(pred_raw)
            prediction_score = float(pred_raw)
            selected_class_idx = target_class_index

        # Align SHAP values with the predicted class
        if isinstance(raw_shap_values, list):
            class_idx = min(selected_class_idx, len(raw_shap_values) - 1)
            shap_vec = raw_shap_values[class_idx][0]
            base_val = (
                self.explainer.expected_value[class_idx]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            class_idx = min(selected_class_idx, raw_shap_values.shape[2] - 1)
            shap_vec = raw_shap_values[0, :, class_idx]
            base_val = self.explainer.expected_value[class_idx]
        else:
            shap_vec = raw_shap_values[0]
            base_val = (
                self.explainer.expected_value[0]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )

        contributions: List[Dict[str, Any]] = []
        for name, s_val in zip(feature_names, shap_vec):
            raw_val = instance_df[name].values[0]
            contributions.append({
                "feature": str(name),
                "actual_value": (
                    float(raw_val)
                    if isinstance(raw_val, (np.number, int, float))
                    else str(raw_val)
                ),
                "shap_value": float(s_val)
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "prediction": prediction_score,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "base_value": float(base_val),
            "feature_contributions": contributions
        }

    def explain_global(self, dataset_df: pd.DataFrame, max_samples: int = 300) -> Dict[str, Any]:
        eval_df = dataset_df.head(max_samples)
        raw_shap = self.explainer.shap_values(eval_df)

        if isinstance(raw_shap, list):
            matrix = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            matrix = raw_shap[:, :, 1]
        else:
            matrix = raw_shap

        mean_abs_impact = np.mean(np.abs(matrix), axis=0)
        global_importance = [
            {"feature": col, "mean_abs_shap": float(val)}
            for col, val in zip(eval_df.columns, mean_abs_impact)
        ]
        global_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        return {
            "samples_evaluated": len(eval_df),
            "global_importance": global_importance
        }