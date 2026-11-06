import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load data as a DataFrame
iris = load_iris(as_frame=True)
df = iris.frame

# 2. Stratified train/test split (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target']
)

# 3. Export training set to CSV (including species label names)
train_export = train_df.copy()
train_export['species'] = train_export['target'].map(lambda idx: iris.target_names[idx])
train_export.to_csv("./datasets/iris_train_dataset.csv", index=False)
print("Training dataset exported to 'iris_train_dataset.csv'.")

# 4. Separate features and target
feature_cols = iris.feature_names
X_train, y_train = train_df[feature_cols], train_df['target']
X_test, y_test = test_df[feature_cols], test_df['target']

# 5. Build Pipeline and GridSearch CV
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__max_depth": [2, 3, 4, None],
    "rf__min_samples_split": [2, 3],
    "rf__criterion": ["gini", "entropy"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 6. Evaluate optimal estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Optimal Parameters: {grid_search.best_params_}")
print(f"Holdout Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Serialize final model pipeline
model_filename = "./models/iris_random_forest_best.joblib"
joblib.dump(best_model, model_filename, compress=3)
print(f"Model saved successfully to '{model_filename}'.")