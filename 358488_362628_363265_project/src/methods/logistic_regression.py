import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """
    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.max_iters = max_iters
        self.weights = None
        self.lr = lr
        
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # Dimensions
        N, D = training_data.shape
        C = get_n_classes(training_labels)

        # Label one-hot-encoding conversion
        labels_one_hot = label_to_onehot(training_labels)
        
        # Random weight initialisation
        self.weights = np.random.normal(0, 0.1, (D, C))
        
        for iteration in range(self.max_iters):
            # Gradient descent
            # Compute softmax probabilities
            curr = np.dot(training_data, self.weights)
            curr = curr - np.max(curr, axis=1, keepdims=True)
            exp_curr = np.exp(curr)
            softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True)
            
            # Compute gradient
            gradient = np.dot(training_data.T, (softmax_probabilities - labels_one_hot))
            
            # Update weights
            self.weights -= self.lr * gradient
        
        # All done: return predicted labels
        pred_labels = self.predict(training_data)
        return pred_labels
        
    def predict(self, test_data):
        """
        Runs prediction on the test data.
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Check if model has been trained
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Compute softmax probabilities
        curr = np.dot(test_data, self.weights)
        curr = curr - np.max(curr, axis=1, keepdims=True)
        exp_curr = np.exp(curr)
        softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True)
        
        # Get class with highest probability
        pred_labels = np.argmax(softmax_probabilities, axis=1)
        return pred_labels