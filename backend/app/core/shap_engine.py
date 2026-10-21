import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional

class ShapExplainerEngine:
    def __init__(self, model: Any, background_data: Optional[pd.DataFrame] = None):
        """
        Initializes SHAP explainer with automated algorithm selection.
        Routes tree-based models to TreeExplainer and general black-boxes to KernelExplainer.
        """
        self.model = model
        self.background_data = background_data
        self.explainer = self._init_explainer()

    def _init_explainer(self):
        try:
            # Fast tree path (RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM)
            return shap.TreeExplainer(self.model)
        except Exception:
            if self.background_data is None or len(self.background_data) == 0:
                raise ValueError("KernelExplainer requires non-empty background evaluation data.")
            
            # Compress background data with k-means to avoid O(2^|F|) computation explosion
            k_samples = min(30, len(self.background_data))
            bg_summary = shap.kmeans(self.background_data, k_samples)
            
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            return shap.KernelExplainer(predict_fn, bg_summary)

    def explain_instance(self, instance_df: pd.DataFrame, target_class_index: int = 1) -> Dict[str, Any]:
        """
        Calculates local feature attribution for a single row.
        Normalizes outputs for both binary classification and continuous regression.
        """
        feature_names = list(instance_df.columns)
        raw_shap_values = self.explainer.shap_values(instance_df)

        # Standardize return signatures across TreeExplainer and KernelExplainer
        if isinstance(raw_shap_values, list):
            selected_idx = min(target_class_index, len(raw_shap_values) - 1)
            shap_vec = raw_shap_values[selected_idx][0]
            base_val = (
                self.explainer.expected_value[selected_idx]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            selected_idx = min(target_class_index, raw_shap_values.shape[2] - 1)
            shap_vec = raw_shap_values[0, :, selected_idx]
            base_val = self.explainer.expected_value[selected_idx]
        else:
            shap_vec = raw_shap_values[0]
            base_val = (
                self.explainer.expected_value[0]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )

        # Compute point prediction
        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(instance_df)
            prediction_score = float(preds[0][target_class_index]) if preds.ndim > 1 else float(preds[0])
        else:
            prediction_score = float(self.model.predict(instance_df)[0])

        # Serialize feature attribution payload
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

        # Sort contributions by absolute SHAP magnitude descending
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "prediction": prediction_score,
            "base_value": float(base_val),
            "feature_contributions": contributions
        }

    def explain_global(self, dataset_df: pd.DataFrame, max_samples: int = 300) -> Dict[str, Any]:
        """
        Computes mean absolute SHAP values across evaluation samples for global importance.
        """
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