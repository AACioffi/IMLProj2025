import numpy as np

def preprocess_data_cnn(train_images, test_images, train_labels, test_labels):
    """
    Preprocesses image data for CNN training.

    1. Normalizes pixel values from [0,255] to [0,1] range.
    2. Transposes data from (N, H, W, C) to (N, C, H, W) for PyTorch.
    3. Creates stratified train/validation split (80/20) maintaining class proportions.
    4. Computes normalized class weights for handling imbalanced dataset.

    Args:
        train_images: Training images array of shape (N, 28, 28, 3)
        test_images:  Test images array of shape (N_test, 28, 28, 3)
        train_labels: Training labels array of shape (N,)
        test_labels:  Test labels array of shape (N_test,)

    Returns:
        train_set_images:         Normalized, transposed training images
        train_set_labels:         Corresponding training labels
        validation_set_images:    Normalized, transposed validation images
        validation_set_labels:    Corresponding validation labels
        proc_test_images:         Normalized, transposed test images
        test_labels:              Unchanged test labels
        normalised_weights:       Class weights array of shape (n_classes,)
    """
    # Normalize
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Transpose to (N, C, H, W)
    train_images = train_images.transpose(0, 3, 1, 2)
    test_images = test_images.transpose(0, 3, 1, 2)

    # Number of classes
    n_classes = len(np.unique(train_labels))
    p_validation = 0.2

    train_images_list = []
    train_labels_list = []
    validation_images_list = []
    validation_labels_list = []

    for class_index in range(n_classes):
        indices = np.where(train_labels == class_index)[0]
        np.random.shuffle(indices)
        n_val = int(len(indices) * p_validation)

        validation_idx = indices[:n_val]
        train_idx = indices[n_val:]

        validation_images_list.append(train_images[validation_idx])
        validation_labels_list.append(train_labels[validation_idx])
        train_images_list.append(train_images[train_idx])
        train_labels_list.append(train_labels[train_idx])

    train_set_images = np.vstack(train_images_list)
    train_set_labels = np.concatenate(train_labels_list)
    validation_set_images = np.vstack(validation_images_list)
    validation_set_labels = np.concatenate(validation_labels_list)

    # Compute class weights
    class_counts = np.bincount(train_set_labels.astype(int))
    total_samples = len(train_set_labels)
    raw_weights = total_samples / class_counts
    normalised_weights = raw_weights * (n_classes / np.sum(raw_weights))

    return (
        train_set_images,
        train_set_labels,
        validation_set_images,
        validation_set_labels,
        test_images,
        test_labels,
        normalised_weights,
    )
