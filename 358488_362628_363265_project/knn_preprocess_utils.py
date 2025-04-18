import numpy as np
from src.utils import normalize_fn

def preprocess_data_knn(X_train, X_test=None):
    # Define feature groups
    categorical_features = {
        'indices': [1, 2, 5, 6, 8, 10, 12],
        'values': [
            [0, 1],         # sex
            [1, 2, 3, 4],   # cp
            [0, 1],         # fbs
            [0, 1, 2],      # restecg
            [0, 1],         # exang
            [1, 2, 3],      # slope
            [3, 6, 7]       # thal
        ]
    }
    numerical_indices = [0, 3, 4, 7, 9, 11]

    # Train
    X_num_train = X_train[:, numerical_indices]
    means = X_num_train.mean(axis=0, keepdims=True)
    stds = X_num_train.std(axis=0, keepdims=True)
    X_num_train_scaled = normalize_fn(X_num_train, means, stds)
    X_cat_train = []

    for i, idx in enumerate(categorical_features['indices']):
        for val in categorical_features['values'][i]:
            bin_col = (X_train[:, idx] == val).astype(float).reshape(-1, 1)
            X_cat_train.append(bin_col)
    X_cat_train = np.hstack(X_cat_train)
    processed_train = np.hstack((X_num_train_scaled, X_cat_train))

    if X_test is not None:
        X_num_test = X_test[:, numerical_indices]
        X_num_test_scaled = normalize_fn(X_num_test, means, stds)
        X_cat_test = []

        for i, idx in enumerate(categorical_features['indices']):
            for val in categorical_features['values'][i]:
                bin_col = (X_test[:, idx] == val).astype(float).reshape(-1, 1)
                X_cat_test.append(bin_col)
        X_cat_test = np.hstack(X_cat_test)
        processed_test = np.hstack((X_num_test_scaled, X_cat_test))
        return processed_train, processed_test

    return processed_train
