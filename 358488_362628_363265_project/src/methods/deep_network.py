import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ### Goal: set the layers of the CNN
        in_height = in_width = 28 # DermaMNIST input size 28x28 (given)

        ## 1) Convolutional Layer
        # kernel_size = k => sliding window of k*k images
        kernel_size = 5
        # stride = k => (k-1) space between each pixel of the sliding window => capturing context
        stride = 1
        # padding => preserve spatial output size (height and width)
        padding = math.floor((kernel_size - stride)/2)
        # out_channels => number of features learned (feature map)
        out_channels1 = 16
        out_channels2 = 2*out_channels1 # after second conv layer
        # first conv => output size: (N, 16, H, W)
        self.conv1 = nn.Conv2d(input_channels, out_channels1, kernel_size=kernel_size, stride=stride, padding=padding)
        # second conv => output: (N, 32, H/2, W/2) (spatial size change because of pooling after conv1)
        # increase feature depth 16 -> 32 (out_channels)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size, stride=stride, padding=padding)


        ## 2) Pooling Layer
        pool_kernel_size = 2
        pool_stride = 2
        # after pooling => downsample by 2 => output size: (N, 16, H/2, W/2)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        # Final spatial size after 2 poolings
        # Each pooling layer halves the height and width
        pooled_height = in_height // (2 ** 2)  # 28 -> 14 -> 7
        pooled_width = in_width // (2 ** 2)


        ## 2) Fully Connected Layer
        # flatten output of conv to feed to activation function and classify input
        # flattened vector size is nb of features times spatial size after conv & pool output
        in_features = out_channels2 * pooled_height * pooled_width  # 32 * 7 * 7 = 1568
        neurons = 128 # = output size of FCL1 = input size of FCL2 = number of patterns recognized and valued
        self.fc1 = nn.Linear(in_features, neurons)
        self.fc2 = nn.Linear(neurons, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ### Goal: make the data flow through the predefined layers

        # applying 2 pooling layers
        #apply activation function ReLu to break linearity of decision boundaries
        x = self.pool(F.relu(self.conv1(x)))  # -> (N, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (N, 32, H/4, W/4)

        # flatten tensor for FCL
        x = x.view(x.size(0), -1)  # flatten except batch dimension (-1 => infer the rest of the dimensions to flatten)

        x = F.relu(self.fc1(x))
        preds = self.fc2(x)  # logits
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = ...  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
