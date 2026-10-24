import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from app.core.inference import predict_frame
from app.core.shap_engine import (
    BackgroundDataRequired,
    ShapExplainerEngine,
    UNITS_LOG_ODDS,
    UNITS_PROBABILITY,
    UNITS_TARGET,
    normalize_shap_output,
)

TOL = 1e-6


@pytest.fixture(scope="module")
def binary():
    data = load_breast_cancer(as_frame=True)
    return data.data.iloc[:, :8], data.target


@pytest.fixture(scope="module")
def iris():
    data = load_iris(as_frame=True)
    return data.data, data.target


@pytest.fixture(scope="module")
def regression():
    data = load_diabetes(as_frame=True)
    return data.data, data.target


def _assert_additive(result):
    total = result["base_value"] + sum(c["shap_value"] for c in result["feature_contributions"])
    assert abs(total - result["explained_output"]) < TOL
    # Interventional TreeExplainer can leave ~1e-3 numerical slack against predict_proba
    assert result["additivity_gap"] < 5e-3


@pytest.mark.parametrize("with_background", [False, True])
def test_random_forest_binary_is_additive_in_probability_space(binary, with_background):
    X, y = binary
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    engine = ShapExplainerEngine(model, "classification", X if with_background else None)
    result = engine.explain_instance(X.iloc[[3]])

    _assert_additive(result)
    assert result["output_units"] == UNITS_PROBABILITY
    assert result["explained_class"] == result["predicted_class"]
    assert result["prediction"] == pytest.approx(max(result["probabilities"].values()))


def test_gradient_boosting_without_background_reports_log_odds(binary):
    X, y = binary
    model = GradientBoostingClassifier(random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", None).explain_instance(X.iloc[[3]])
    _assert_additive(result)
    assert result["output_units"] == UNITS_LOG_ODDS


def test_gradient_boosting_with_background_reports_probability(binary):
    X, y = binary
    model = GradientBoostingClassifier(random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", X).explain_instance(X.iloc[[3]])
    _assert_additive(result)
    assert result["output_units"] == UNITS_PROBABILITY
    assert result["explained_output"] == pytest.approx(result["prediction"], abs=1e-4)


def test_single_output_binary_explains_the_predicted_negative_class(binary):
    X, y = binary
    model = GradientBoostingClassifier(random_state=0).fit(X, y)
    negative_row = X[model.predict(X) == 0].iloc[[0]]
    result = ShapExplainerEngine(model, "classification", X).explain_instance(negative_row)
    assert result["predicted_class"] == "0"
    assert result["explained_class"] == "0"
    assert result["prediction"] > 0.5
    _assert_additive(result)


def test_kernel_explainer_uses_background_and_is_not_degenerate(iris):
    X, y = iris
    model = LogisticRegression(max_iter=1000).fit(X, y)
    engine = ShapExplainerEngine(model, "classification", X, kmeans_k=10)
    assert engine.explainer_type.startswith("KernelExplainer")
    result = engine.explain_instance(X.iloc[[0]])
    _assert_additive(result)
    # Regression test: a single-row background used to yield all-zero attributions
    assert any(abs(c["shap_value"]) > 1e-3 for c in result["feature_contributions"])


def test_kernel_explainer_without_background_is_rejected(iris):
    X, y = iris
    model = LogisticRegression(max_iter=1000).fit(X, y)
    with pytest.raises(BackgroundDataRequired):
        ShapExplainerEngine(model, "classification", None)


def test_regressor_reports_target_units(regression):
    X, y = regression
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "regression", None).explain_instance(X.iloc[[5]])
    _assert_additive(result)
    assert result["output_units"] == UNITS_TARGET
    assert result["predicted_class"] is None
    assert result["prediction"] == pytest.approx(float(model.predict(X.iloc[[5]])[0]))


def test_multiclass_explains_predicted_class(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    row = X.iloc[[120]]
    result = ShapExplainerEngine(model, "classification", X).explain_instance(row)
    _assert_additive(result)
    assert result["explained_class"] == result["predicted_class"] == str(model.predict(row)[0])


def test_predict_and_explain_agree(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    row = X.iloc[[60]]
    pred = predict_frame(model, "classification", row)
    expl = ShapExplainerEngine(model, "classification", X).explain_instance(row)
    assert pred["predicted_class"] == expl["predicted_class"]
    assert pred["prediction"] == pytest.approx(expl["prediction"])
    assert pred["probabilities"] == pytest.approx(expl["probabilities"])


def test_contributions_sorted_by_absolute_impact(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", None).explain_instance(X.iloc[[0]])
    magnitudes = [abs(c["shap_value"]) for c in result["feature_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_global_multiclass_averages_classes_and_returns_beeswarm(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", X).explain_global(X, max_samples=60)

    assert result["samples_evaluated"] == 60
    assert "averaged" in result["explained_output"]
    importances = [g["mean_abs_shap"] for g in result["global_importance"]]
    assert importances == sorted(importances, reverse=True)
    bees = result["beeswarm"]
    assert bees["features"] == [g["feature"] for g in result["global_importance"]]
    assert len(bees["shap_values"]) == X.shape[1]
    assert all(len(col) == 60 for col in bees["shap_values"] + bees["feature_values"])


def test_global_specific_class(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", X).explain_global(X, max_samples=30, class_index=2)
    assert result["explained_output"] == "class 2"


def test_global_kernel_is_capped(iris):
    X, y = iris
    model = LogisticRegression(max_iter=1000).fit(X, y)
    engine = ShapExplainerEngine(model, "classification", X, kmeans_k=5)
    assert engine.explain_global(X, max_samples=300)["samples_evaluated"] <= 50


@pytest.mark.parametrize("raw, shape", [
    (np.zeros((3, 4)), (3, 4, 1)),
    (np.zeros(4), (1, 4, 1)),
    ([np.zeros((3, 4)), np.zeros((3, 4))], (3, 4, 2)),
    (np.zeros((3, 4, 5)), (3, 4, 5)),
])
def test_normalize_shap_output_shapes(raw, shape):
    assert normalize_shap_output(raw).shape == shape


def test_string_feature_values_are_preserved(iris):
    X, y = iris
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    result = ShapExplainerEngine(model, "classification", None).explain_instance(X.iloc[[0]])
    assert all(isinstance(c["actual_value"], float) for c in result["feature_contributions"])
    assert isinstance(pd.DataFrame(result["feature_contributions"]), pd.DataFrame)
