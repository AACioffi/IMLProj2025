import numpy as np
from collections import Counter

K_OPTIMAL = 48

class KMeans(object):
    """
    kMeans classifier object.
    """

    def __init__(self, max_iters=500, k=5):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None
        self.k = k

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
        # number of clusters
        #k = len(np.unique(training_labels))
        #k = self.k
        k = K_OPTIMAL
        N = training_data.shape[0]

        rd_index = np.random.choice(N, size=k, replace=True)
        self.centroids = training_data[rd_index] #Initializing the k centroids to k random samples
        pred_centroids = np.zeros(N, dtype=int)
        for iteration in range(self.max_iters):
            for i in range(N):
                distances = np.zeros(k)
                for j in range(k):
                    distances[j] = np.linalg.norm(training_data[i] - self.centroids[j])
                pred_centroids[i] = np.argmin(distances)
            for i in range(k):
                assigned = training_data[pred_centroids == i] #extracts all samples assigned to centroid i
                if len(assigned) == 0:
                    continue  # Skip update for empty cluster if data is not sufficient
                self.centroids[i] = assigned.mean(axis=0)

        self.best_permutation = np.zeros(k)
        for i in range(k):
            cluster_labels = training_labels[pred_centroids == i] #extracts all labels contained in cluster i
            # Finds most common label in the cluster i and assigns it to best_permutation at index i
            if len(cluster_labels) == 0:
                continue  # Skip empty cluster if data is not sufficient
            self.best_permutation[i] = Counter(cluster_labels).most_common(1)[0][0]

        pred_labels = np.zeros(N)
        for i in range(N):
            pred_labels[i] = self.best_permutation[pred_centroids[i]]

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
            test_labels[i] = self.best_permutation[np.argmin(distances)]

        return test_labels
