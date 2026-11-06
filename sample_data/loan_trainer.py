import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Synthesize lending & credit underwriting dataset (LendingClub schema)
np.random.seed(42)
n_samples = 2500

annual_inc = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples).round(-2)
fico_score = np.random.normal(loc=700, scale=45, size=n_samples).clip(580, 850).round()
loan_amnt = np.random.uniform(3000, 35000, size=n_samples).round(-2)
int_rate = (28.0 - (fico_score - 580) * 0.06 + np.random.normal(0, 1.5, size=n_samples)).clip(5.5, 29.9).round(2)
dti = (np.random.gamma(shape=3.0, scale=6.0, size=n_samples)).clip(2.0, 45.0).round(2)
revol_util = (np.random.beta(a=2, b=2, size=n_samples) * 100).round(1)
total_acc = np.random.poisson(lam=22, size=n_samples).clip(4, 60)
pub_rec = np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03], size=n_samples)
revol_bal = (annual_inc * np.random.uniform(0.05, 0.35, size=n_samples)).round()

risk_logit = (
    -3.5
    + 0.12 * int_rate
    + 0.06 * dti
    + 0.015 * revol_util
    + 0.45 * pub_rec
    - 0.008 * (fico_score - 600)
    + 0.00003 * loan_amnt
    - 0.000008 * annual_inc
)
prob_default = 1 / (1 + np.exp(-risk_logit))
loan_status = np.random.binomial(1, prob_default)

feature_cols = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti', 
    'fico_score', 'revol_util', 'revol_bal', 'total_acc', 'pub_rec'
]

df = pd.DataFrame({
    'loan_amnt': loan_amnt,
    'int_rate': int_rate,
    'annual_inc': annual_inc,
    'dti': dti,
    'fico_score': fico_score,
    'revol_util': revol_util,
    'revol_bal': revol_bal,
    'total_acc': total_acc,
    'pub_rec': pub_rec,
    'loan_status': loan_status
})

# 2. Stratified train/test split (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['loan_status']
)

X_train_raw = train_df[feature_cols]
y_train = train_df['loan_status']
X_test_raw = test_df[feature_cols]
y_test = test_df['loan_status']

# 3. Fit scaler separately & retain column headers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Reconstruct DataFrames to keep feature names intact for SHAP explainers
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)

# 4. Save background dataset and inference test set in the scaled feature space
# (Essential: C-Box expects background dataset features to be in the exact numerical space the model accepts)
train_background = X_train_scaled_df.copy()
train_background['loan_status'] = y_train.values
train_background['loan_status_label'] = train_background['loan_status'].map({0: 'Fully Paid', 1: 'Charged Off'})
train_background.to_csv("./datasets/financial_credit_risk_scaled_train_background.csv", index=False)

test_features_export = X_test_scaled_df.copy().reset_index(drop=True)
test_features_export.to_csv("./datasets/financial_credit_risk_scaled_test_features.csv", index=False)

# 5. Fit standalone RandomForestClassifier directly on scaled data
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=4,
    random_state=42
)
rf.fit(X_train_scaled_df, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test_scaled_df)
y_prob = rf.predict_proba(X_test_scaled_df)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
print(classification_report(y_test, y_pred, target_names=['Fully Paid', 'Charged Off']))

# 7. Serialize standalone model and scaler separately
joblib.dump(rf, "./models/financial_random_forest.joblib", compress=3)
print("Saved standalone model to 'financial_random_forest.joblib'.")