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
        self.weights = None  # Sample weights?
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
        labels_one_hot = label_to_onehot(training_labels)
        
        # Random weight initialisation
        self.weights = np.random.normal(0, 0.1, (D, C))

        ### DEBUG: Tracking lists for diagnostics ###
        #loss_history = []
        #gradient_norm_history = []
        #weights_norm_history = []
        #############################################
        
        for iteration in range(self.max_iters):
            # Gradient descent
            
            # Compute softmax probabilities
            curr = np.dot(training_data, self.weights)
            curr = curr - np.max(curr, axis=1, keepdims=True)  # For numerical stability
            exp_curr = np.exp(curr)
            softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True)
            
            ### DEBUG: Compute loss (cross-entropy) ####################################
            #epsilon = 1e-15
            #loss = -np.sum(labels_one_hot * np.log(softmax_probabilities + epsilon)) / N
            ############################################################################
            
            # Compute error
            error = softmax_probabilities - labels_one_hot
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Reshape sample_weights to apply to each class prediction for each sample
                sample_weights_reshaped = sample_weights.reshape(-1, 1)
                # Weight the errors by sample importance
                weighted_error = error * sample_weights_reshaped
                # Compute weighted gradient
                gradient = np.dot(training_data.T, weighted_error)
            else:
                # Standard gradient computation without weights
                gradient = np.dot(training_data.T, error)


            ### DEBUG: Track diagnostic information #################
            #loss_history.append(loss)
            #gradient_norm_history.append(np.linalg.norm(gradient))
            #weights_norm_history.append(np.linalg.norm(self.weights))
            #########################################################
            
            # Update weights
            self.weights -= self.lr * gradient

            ### DEBUG: print diagnostics every 10 iterations ############
            #if iteration % 10 == 0:
            #    print(f"Iteration {iteration}:")
            #    print(f"  Loss: {loss}")
            #    print(f"  Gradient Norm: {np.linalg.norm(gradient)}")
            #    print(f"  Weights Norm: {np.linalg.norm(self.weights)}")
            #############################################################

        ### DEBUG: Print final diagnostic information ###################
        #print("\nTraining Diagnostics:")
        #print("Final Loss:", loss_history[-1])
        #print("Final Gradient Norm:", gradient_norm_history[-1])
        #print("Final Weights Norm:", weights_norm_history[-1])
        #################################################################
        
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
        curr = curr - np.max(curr, axis=1, keepdims=True)  # For numerical stability
        exp_curr = np.exp(curr)
        softmax_probabilities = exp_curr / np.sum(exp_curr, axis=1, keepdims=True)

        ### DEBUG: Diagnostic prints ########################################
        #print("Softmax probabilities shape:", softmax_probabilities.shape)
        #print("Softmax probabilities min:", np.min(softmax_probabilities))
        #print("Softmax probabilities max:", np.max(softmax_probabilities))
        #print("Any NaNs in softmax:", np.isnan(softmax_probabilities).any())
        #####################################################################
        
        # Get class with highest probability
        pred_labels = np.argmax(softmax_probabilities, axis=1)

        ### DEBUG: Additional diagnostic ########################
        #print("Predicted labels:", pred_labels)
        #print("Unique predicted labels:", np.unique(pred_labels))
        #########################################################

        # Ensure predictions are float to match original label type
        pred_labels = pred_labels.astype(float)
        
        return pred_labels