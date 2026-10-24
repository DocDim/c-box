import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load canonical Iris dataset (150 rows, 4 features)
iris = load_iris(as_frame=True)
df_original = iris.frame

# -------------------------------------------------------------
# Method 1: Bootstrapping / Oversampling with Noise (Jittering)
# -------------------------------------------------------------
# Samples 1,000 instances with replacement and adds minor Gaussian
# jitter to avoid exact duplicate rows during gradient descent/training.
np.random.seed(42)
sampled_df = df_original.sample(n=1000, replace=True).reset_index(drop=True)

feature_cols = iris.feature_names
noise = np.random.normal(loc=0, scale=0.02, size=(1000, len(feature_cols)))
sampled_df[feature_cols] = (sampled_df[feature_cols] + noise).round(2)

# Save to CSV
sampled_df.to_csv("iris_1000_samples.csv", index=False)


# -------------------------------------------------------------
# Method 2: Parametric Synthetic Generation (Gaussian Sampling)
# -------------------------------------------------------------
# Preserves class covariance, variances, and feature correlations.
def generate_synthetic_iris(df, target_col="target", samples_per_class=334):
    synthetic_records = []
    classes = df[target_col].unique()

    for cls in classes:
        cls_data = df[df[target_col] == cls].drop(columns=[target_col])
        mean = cls_data.mean().values
        cov = cls_data.cov().values

        synth_features = np.random.multivariate_normal(
            mean, cov, size=samples_per_class
        )
        synth_df = pd.DataFrame(synth_features, columns=cls_data.columns)
        synth_df[target_col] = cls
        synthetic_records.append(synth_df)

    combined = pd.concat(synthetic_records, ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # Shuffle


df_synthetic = generate_synthetic_iris(df_original, samples_per_class=334).iloc[
    :1000
]
df_synthetic[feature_cols] = df_synthetic[feature_cols].round(2)