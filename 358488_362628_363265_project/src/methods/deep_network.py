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
    def __init__(self, input_size, n_classes, n1=384, n2=384):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        self.l1 = nn.Linear(input_size, n1)
        self.l2 = nn.Linear(n1, n2)
        self.l3 = nn.Linear(n2, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.dropout(F.gelu(self.l1(x)), p=0.2)
        x = F.dropout(F.gelu(self.l2(x)), p=0.2)
        preds = self.l3(x)  # logits
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

        self.bn1   = nn.BatchNorm2d(out_channels1) #try with batch normalization

        # second conv => output: (N, 32, H/2, W/2) (spatial size change because of pooling after conv1)
        # increase feature depth 16 -> 32 (out_channels)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn2   = nn.BatchNorm2d(out_channels2) #try with batch normalization


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

        self.dropout = nn.Dropout(p=0.5) #try with dropout

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (N, 16, H/2, W/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (N, 32, H/4, W/4)

        # flatten tensor for FCL
        x = x.view(x.size(0), -1)  # flatten except batch dimension (-1 => infer the rest of the dimensions to flatten)

        x = F.relu(self.fc1(x))
        x = self.dropout(x) #try with dropout
        preds = self.fc2(x)  # logits
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, class_weights=None):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
            class_weights (array): class weights, used to handle class imbalance
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep=None):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()

        # Tracking variables
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(torch.float32)
            batch_y = batch_y.to(torch.long)

            # Forward pass
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            # Backprop + optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

             # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
        if ep is not None and ep % 10 == 0:  # Print every 10 epochs
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {ep}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                # batch might be a tuple (data,)
                batch_x = batch[0]
                logits = self.model(batch_x.to(torch.float32))
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)

        pred_labels = torch.cat(all_preds, dim=0)
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