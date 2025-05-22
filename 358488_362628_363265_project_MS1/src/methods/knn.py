import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels
        
        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        num_test = test_data.shape[0]
        test_labels = np.empty(num_test, dtype=self.training_labels.dtype)

        for i in range(num_test):
            # Compute L2 distances to all training points
            distances = np.linalg.norm(self.training_data - test_data[i], axis=1)

            # Get indices of k nearest neighbors
            neighbor_idxs = np.argsort(distances)[:self.k]

            # Get neighbor labels
            neighbor_labels = self.training_labels[neighbor_idxs]

            if self.task_kind == "classification":
                # Majority vote
                test_labels[i] = np.bincount(neighbor_labels.astype(int)).argmax()
            else:
                # For regression, take the mean
                test_labels[i] = np.mean(neighbor_labels)

        return test_labels