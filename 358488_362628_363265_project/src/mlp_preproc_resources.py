import numpy as np
from data import load_data


# Load and inspect raw data
train_images, test_images, train_labels, test_labels = load_data()

# Checking dimensions
N_tr, H_tr, W_tr, C_tr = train_images.shape
N_te, H_te, W_te, C_te = test_images.shape

print(f'Train images shape -- N_tr = {N_tr}, H_tr = {H_tr}, W_tr = {W_tr}, C_tr = {C_tr}')
print(f'Test images shape  -- N_te = {N_te}, H_te = {H_te}, W_te = {W_te}, C_te = {C_te}')

# Loading data to examine classes
extracted_classes = np.unique(train_labels)
n_classes = len(extracted_classes)

class_counts = np.bincount(train_labels.astype(int))

print(f'Classes:')
for c in extracted_classes:
    print(f'{c}')
print(f'Number of classes: {n_classes}')
print(f'Class counts: {class_counts}')


def preprocess_data_mlp(train_images, test_images, train_labels, test_labels):
    """
    Preprocesses image data for MLP training

    Performs the following operations:
    1. Flattens 28x28x3 images into 1D vectors (2352 features)
    2. Normalizes pixel values from [0,255] to [0,1] range
    3. Creates stratified train/validation split (80/20) maintaining class proportions
    4. Calculates normalized class weights for handling imbalanced dataset

    Args:
        train_images: Training images array of shape (N, 28, 28, 3)
        test_images:  Test images array of shape (N_test, 28, 28, 3)
        train_labels: Training labels array of shape (N,)
        test_labels:  Test labels array of shape (N_test,)

    Returns:
        train_set_datapoints:      Flattened, normalized training data (80% of original)
        train_set_labels:          Corresponding training labels
        validation_set_datapoints: Flattened, normalized validation data (20% of original)
        validation_set_labels:     Corresponding validation labels
        proc_test_images:          Flattened, normalized test data
        test_labels:               Original test labels (unchanged)
        normalised_weights:        Class weights for loss function (shape: n_classes,)
    """
    # Image dimensions
    N_tr, H_tr, W_tr, C_tr = train_images.shape
    N_te, H_te, W_te, C_te = test_images.shape

    # Flatten images
    proc_train_images = train_images.reshape(N_tr, -1).astype(np.float32)
    proc_test_images = test_images.reshape(N_te, -1).astype(np.float32)

    # Normalize pixel values to [0,1]
    proc_train_images /= 255.0
    proc_test_images /= 255.0

    # Number of classes
    n_classes = len(np.unique(train_labels))

    # Stratified split (80/20)
    p_validation = 0.2

    train_images_list = []
    train_labels_list = []
    validation_images_list = []
    validation_labels_list = []

    for class_index in range(n_classes):
        class_indices = np.where(train_labels == class_index)[0]
        np.random.shuffle(class_indices)
        n_validation = int(len(class_indices) * p_validation)

        validation_indices = class_indices[:n_validation]
        training_indices = class_indices[n_validation:]

        validation_images_list.append(proc_train_images[validation_indices])
        validation_labels_list.append(train_labels[validation_indices])
        train_images_list.append(proc_train_images[training_indices])
        train_labels_list.append(train_labels[training_indices])

    train_set_datapoints = np.vstack(train_images_list)
    train_set_labels = np.concatenate(train_labels_list)
    validation_set_datapoints = np.vstack(validation_images_list)
    validation_set_labels = np.concatenate(validation_labels_list)

    # Class weights for imbalance
    class_counts = np.bincount(train_set_labels.astype(int))
    total_samples = len(train_set_labels)
    raw_weights = total_samples / class_counts
    normalised_weights = raw_weights * (n_classes / np.sum(raw_weights))

    return (
        train_set_datapoints,
        train_set_labels,
        validation_set_datapoints,
        validation_set_labels,
        proc_test_images,
        test_labels,
        normalised_weights,
    )
