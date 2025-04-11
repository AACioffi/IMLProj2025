import numpy as np
import itertools


class KMeans(object):
    """
    kMeans classifier object.
    """

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        k = len(np.unique(training_labels)) #number of clusters
        D = training_data.shape[1] #number of features
        self.centroids = np.zeros((k, D)) #centroids initialization
        pred_labels = np.zeros(training_data.shape[0])

        for iteration in range(self.max_iters):
            for i in range(training_data.shape[0]):
                distances = np.zeros(k)
                for j in range(k):
                    distances[j] = np.linalg.norm(training_data[i] - self.centroids[j])
                pred_labels[i] = np.argmin(distances)
            for i in range(k):
                assigned = training_data[pred_labels == i] #extracts all samples assigned to centroid i
                self.centroids[i] = assigned.mean(axis=0)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]
        test_labels = np.zeros(N)
        k = len(self.centroids)

        for i in range(N):
            distances = np.zeros(k)
            for j in range(k):
                distances[j] = np.linalg.norm(test_data[i] - self.centroids[j])
            test_labels[i] = np.argmin(distances)

        return test_labels
