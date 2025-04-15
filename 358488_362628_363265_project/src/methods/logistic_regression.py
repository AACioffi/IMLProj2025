import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label

class LogisticRegression(object):
    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.max_iters = max_iters
        self.weights = None # Will change during execution of the algorithm
        self.lr = lr
        
    def fit(self, training_data, training_labels, sample_weights=None):
        """
        Trains the model, returns predicted labels for training data.
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
            sample_weights (array, optional): weights of shape (N,) for each sample
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # Dimensions
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        
        # Label one-hot-encoding conversion
        labels_one_hot = label_to_onehot(training_labels) # shape (N, C)
        
        # Random weight initialisation
        self.weights = np.random.normal(0, 0.1, (D, C)) # Mean 0, std 0.1, shape (D,C)

        for iteration in range(self.max_iters):
            
            # Compute softmax probabilities
            curr = np.dot(training_data, self.weights)         # Shape (N, C) - for each sample, assign a "current" score for each of the C classes (hence curr)
                                                               # Recall: a score tied to a class is the signed distance of the sample from the decision boundary
                                                               # of the considered class
            curr = curr - np.max(curr, axis=1, keepdims=True)  # For numerical stability - find max value of each sample (over rows, as axis = 1),
                                                               # keep it as a column (keepdims = True); for each score in a given row, subtract the
                                                               # value at the same row in the max value column
            exp_curr = np.exp(curr)                            # Element wise exp function application
            softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True) # Element wise division of exp_curr matrix
                                                                                       # The divisor is a column - a given element of said column is the sum of
                                                                                       # all the elements on a row
                                                                                       # The divisor of row i will then be the element at row i in this column
                                                                                       # We get the probability of each class for a given sample (as a row)
                                                                                       # after the element-wise division
            # Compute error
            error = softmax_probabilities - labels_one_hot     # Difference between prediction and true label (still (N, C))
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Reshape sample_weights to apply to each class prediction for each sample
                sample_weights_reshaped = sample_weights.reshape(-1, 1) # Goes from (N, ) to (N, 1) (necessary for broadcast below)
                # Weight the errors by sample importance
                weighted_error = error * sample_weights_reshaped        # Element-wise multiplication (!) of each elem in "error" with sample weights (broadcast)
                                                                        # Results in (N, C) array
                # Compute weighted gradient
                gradient = np.dot(training_data.T, weighted_error)      # (D, N) @ (N, C) - Dimensions match!
            else:
                # Standard gradient computation without weights
                gradient = np.dot(training_data.T, error)               # (D, N) @ (N, C) as well

            # Update weights (gradient descent)
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
        
        # Same as for fit, final softmax probability computation
        curr = np.dot(test_data, self.weights)
        curr = curr - np.max(curr, axis=1, keepdims=True)
        exp_curr = np.exp(curr)
        softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True)
        
        # Get class with highest probability
        pred_labels = np.argmax(softmax_probabilities, axis=1) # Done over rows still...

        # Ensure predictions are float to match original label type
        pred_labels = pred_labels.astype(float)
        
        return pred_labels