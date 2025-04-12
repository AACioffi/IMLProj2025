import numpy as np

def preprocess_data_logireg(X_train, X_test=None):
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
    # Integer types: subtract expectation, divide by standard deviation
    X_numerical_train = X_train[:, numerical_features['numerical_features_indices']]
    X_train_mean = np.mean(X_numerical_train, axis=0)
    X_train_std = np.std(X_numerical_train, axis=0)
    X_numerical_train_scaled = (X_numerical_train - X_train_mean) / X_train_std
    
    # Initiate returned values to the numerical processed values
    processed_X_train = X_numerical_train_scaled
    
    # One-hot encoding process for categorical values
    for i, idx in enumerate(categorical_features['categorical_features_indices']):
        # Get the corresponding [a, b, c, ...] in the categorical feature values
        possible_values = categorical_features['categorical_features_values'][i]
        
        # Check if the categorical feature is binary
        if len(possible_values) == 2:
            # Create just one column (dimensionality reduction)
            # Note: bitmask is trivial, just take one of the possible values as reference (e.g. possible_values[0])
            bin_values_train = (X_train[:, idx] == possible_values[0]).astype(float).reshape(-1, 1)
            processed_X_train = np.hstack((processed_X_train, bin_values_train))
        else:
            # For non-binary features, create a column for each possible value
            for value in possible_values:
                bin_values_train = (X_train[:, idx] == value).astype(float).reshape(-1, 1)
                processed_X_train = np.hstack((processed_X_train, bin_values_train))
    
    # If test data is provided, preprocess it using training data statistics
    if X_test is not None:
        # Same numerical scaling as before
        X_numerical_test = X_test[:, numerical_features['numerical_features_indices']]
        X_numerical_test_scaled = (X_numerical_test - X_train_mean) / X_train_std
        processed_X_test = X_numerical_test_scaled
        
        # Same categorical one-hot encoding as before
        for i, idx in enumerate(categorical_features['categorical_features_indices']):
            possible_values = categorical_features['categorical_features_values'][i]
            
            if len(possible_values) == 2:
                bin_values_test = (X_test[:, idx] == possible_values[0]).astype(float).reshape(-1, 1)
                processed_X_test = np.hstack((processed_X_test, bin_values_test))
            else:
                for value in possible_values:
                    bin_values_test = (X_test[:, idx] == value).astype(float).reshape(-1, 1)
                    processed_X_test = np.hstack((processed_X_test, bin_values_test))
        
        return processed_X_train, processed_X_test
    
    return processed_X_train


def calculate_sample_weights_logireg(y_train):
    # Ensure labels are integers
    y_train = y_train.astype(int)
    
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    
    #print("Unique classes:", np.unique(y_train))
    #print("Class counts:", class_counts)
    
    class_frequencies = class_counts / total_samples
    #print("Class frequencies:", class_frequencies)
    
    # Compute weights (inverse of frequency)
    class_weights = 1 / class_frequencies
    #print("Initial class weights:", class_weights)
    
    # Weight normalisation: they sum to the number of samples
    class_weights = (class_weights / np.sum(class_weights)) * total_samples
    #print("Normalized class weights:", class_weights)
    
    # Mapping each sample to its corresponding class weight
    sample_weights = np.array([class_weights[int(label)] for label in y_train])
    
    #print("Sample weights statistics:")
    #print("  Min:", np.min(sample_weights))
    #print("  Max:", np.max(sample_weights))
    #print("  Mean:", np.mean(sample_weights))
    #print("  Sum:", np.sum(sample_weights))
    
    return sample_weights
