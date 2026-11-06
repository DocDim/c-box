import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. Load reference dataset metadata
iris = load_iris()
feature_names = iris.feature_names
X = iris.data
y = iris.target

# 2. Generate biologically consistent synthetic samples
# Sampling via multivariate Gaussian around each class centroid
np.random.seed(42)
num_samples_per_class = 20
synthetic_records = []

for target_class in [0, 1, 2]:
    class_data = X[y == target_class]
    mean_vec = class_data.mean(axis=0)
    cov_matrix = np.cov(class_data, rowvar=False)
    
    # Slight contraction (0.8) to keep generated points well inside natural boundaries
    sampled = np.random.multivariate_normal(mean_vec, cov_matrix * 0.8, size=num_samples_per_class)
    synthetic_records.append(sampled)

# Stack and round to 1 decimal place matching empirical Iris measurements
synthetic_matrix = np.vstack(synthetic_records).round(1)

# Enforce physical biological boundaries [min, max]
lower_bounds = [4.0, 1.8, 0.8, 0.1]
upper_bounds = [8.5, 4.6, 7.2, 2.8]
synthetic_matrix = np.clip(synthetic_matrix, a_min=lower_bounds, a_max=upper_bounds)

# 3. Create DataFrame with identical feature columns
inference_df = pd.DataFrame(synthetic_matrix, columns=feature_names)

# Export to CSV for batch upload
inference_df.to_csv("./datasets/iris_inference_samples.csv", index=False)
print("Saved 15 synthetic test samples to 'iris_inference_samples.csv'.\n")
print(inference_df.head(6))
