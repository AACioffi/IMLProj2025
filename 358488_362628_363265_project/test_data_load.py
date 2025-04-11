
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


# Function concept for data preprocessing
# Work in progress!
# %%
# Function concept for data preprocessing
# Work in progress!
def preprocess_data(X_train):
    # Feature values as seen in the block above
    categorical_features = {
        'categorical_features_indices': [1, 2, 5, 6, 8, 10, 12],
        'categorical_features_names': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'],
        'categorical_features_values': [
            [0, 1],  # sex values
            [1, 2, 3, 4],  # cp values
            [0, 1],  # fbs values
            [0, 1, 2],  # restecg values
            [0, 1],  # exang values
            [1, 2, 3],  # slope values
            [3, 6, 7]  # thal values
        ]
    }

    numerical_features = {
        'numerical_features_indices': [0, 3, 4, 7, 9, 11],
        'numerical_features_names': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    }

    # Processing
    # Integer types: as seen in exercise series: subtract expectation, divide by standard deviation
    X_numerical_train = X_train[:, numerical_features['numerical_features_indices']]
    X_train_mean = np.mean(X_numerical_train, axis=0)
    X_train_std = np.std(X_numerical_train, axis=0)
    X_numerical_train_scaled = (X_numerical_train - X_train_mean) / X_train_std

    # Now we have arrays containing preprocessed integer types - they should be "merged" with categorical features somehow.

    # Categorical types: a reasonable course of action is to do one-hot encoding
    # This would ensure there are less/no problems with predictions skewing towards greater values (in abs. value)

    X_categorical_train = X_train[:, categorical_features['categorical_features_indices']]

    # Initiate returned values to the numerical processed values
    processed_X_train = X_numerical_train_scaled

    # One-hot encoding process for categorical values

    # Create tuple list [(0, 1), (1, 2), (2, 5), ...] where the first element is the index i, and the second is the dataset index idx taken from
    # the element categorical_features_indices
    for i, idx in enumerate(categorical_features['categorical_features_indices']):
        # Get the corresponding [a, b, c, ...] in the categorical feature values
        possible_values = categorical_features['categorical_features_values'][i]

        # One-hot encoding creation
        for value in possible_values:
            # Create binary features (1 if the feature equals this value, 0 otherwise), by iterating over every possible value and applying mask
            # Done column per column! TODO: For binary categories, only create one column!
            bin_values_train = (X_train[:, idx] == value).astype(float).reshape(-1, 1)

            # Horizontally stacking the processed column to create one-hot encoding for categorical feature at idx
            processed_X_train = np.hstack((processed_X_train, bin_values_train))

    return processed_X_train


# The given dataset contains unbalanced counts belonging to each class.
# We should equalise their impact in some way - as seen in k-NN, we normalise
# using a weight value computed from the frequency of each class.
# Quote k-NN slide 35: "Weighing neighbors by the inverse of their class size converts neighbor counts
# into the fraction of each class that falls in your K nearest neighbors."
def calculate_sample_weights(y_train):
    # Class frequency computation
    classes = np.unique(y_train)  # Should be [0.0, 1.0, 2.0, 3.0, 4.0]
    n_samples = len(y_train)
    n_classes = len(classes)

    # [0, 1, 2, 3, 4] yields counts [128, 41, 30, 30, 8] (see above, first cell)
    class_counts = np.bincount(y_train.astype(int))

    # Calculate weights inversely proportional to class frequency
    # Rare classes get higher weights
    class_weights = n_samples / class_counts  # elem-wise division

    # Map each sample to its corresponding class weight
    sample_weights = np.array([class_weights[int(label)] for label in y_train])

    return sample_weights