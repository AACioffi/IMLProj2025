import numpy as np

def preprocess_data_kmeans(X):
    categorical_indices = [1, 2, 5, 6, 8, 10, 12]
    categorical_values = [
        [0, 1],  # sex
        [1, 2, 3, 4],  # cp
        [0, 1],  # fbs
        [0, 1, 2],  # restecg
        [0, 1],  # exang
        [1, 2, 3],  # slope
        [3, 6, 7]  # thal
    ]
    numerical_indices = [0, 3, 4, 7, 9, 11]

    # Normalize numerical features
    xnum = X[:, numerical_indices]
    means = np.mean(xnum, axis=0)
    stds = np.std(xnum, axis=0)
    xnum_scaled = (xnum - means) / stds

    # One-hot encode categorical features (including binary)
    xcat = []
    for i, idx in enumerate(categorical_indices):
        for val in categorical_values[i]:
            bin_col = (X[:, idx] == val).astype(float).reshape(-1, 1)
            xcat.append(bin_col)
    xcat_encoded = np.hstack(xcat)

    # Combine numerical and categorical
    xprocessed = np.hstack((xnum_scaled, xcat_encoded))
    return xprocessed