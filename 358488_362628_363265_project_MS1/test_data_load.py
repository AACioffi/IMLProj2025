
import numpy as np

# Load the dataset
data = np.load('features.npz')
xtrain = data['xtrain']
ytrain = data['ytrain']

# Check the distinct values in each categorical column
categorical_cols = [1, 2, 5, 6, 8, 10, 12]  # Indices for categorical features
col_names = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for i, col_idx in enumerate(categorical_cols):
    unique_vals = np.unique(xtrain[:, col_idx])
    print(f"{col_names[i]}: {unique_vals}")

# Print counts for each label
values, counts = np.unique(ytrain, return_counts=True)
print("Sample count for each label:")
for i, c in zip(values, counts):
    print(f"{i}, {c}")