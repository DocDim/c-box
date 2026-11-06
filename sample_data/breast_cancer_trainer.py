import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load clinical dataset
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
feature_cols = list(cancer.feature_names)

# 2. Stratified train/test split (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"]
)

# 3. Export reference background dataset for SHAP (with target & diagnosis labels)
train_export = train_df.copy()
train_export["diagnosis"] = train_export["target"].map({0: "malignant", 1: "benign"})
train_export.to_csv("./datasets/breast_cancer_train_background.csv", index=False)
print("Saved background dataset: 'breast_cancer_train_background.csv'")

# 4. Export holdout test set (strictly features, for batch inference)
test_features = test_df[feature_cols].copy().reset_index(drop=True)
test_features.to_csv("./datasets/breast_cancer_test_features.csv", index=False)
print("Saved inference test features: 'breast_cancer_test_features.csv'")

# 5. Build Pipeline: StandardScaler + GradientBoostingClassifier
X_train, y_train = train_df[feature_cols], train_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(random_state=42))
])

# 6. Hyperparameter tuning optimized for ROC-AUC
param_grid = {
    "gb__n_estimators": [50, 100, 150],
    "gb__learning_rate": [0.05, 0.1],
    "gb__max_depth": [2, 3],
    "gb__subsample": [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 7. Validation metrics
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"Optimal Parameters: {grid_search.best_params_}")
print(f"Holdout ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

# 8. Serialize trained pipeline
model_path = "./models/breast_cancer_gb_model.joblib"
joblib.dump(best_model, model_path, compress=3)
print(f"Model saved successfully to '{model_path}'.")