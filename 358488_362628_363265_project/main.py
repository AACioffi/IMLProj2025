import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from src.mlp_preproc_resources import preprocess_data_mlp
from src.cnn_preproc_resources import preprocess_data_cnn

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain_raw, xtest_raw, ytrain_raw, ytest_raw = load_data()

    if args.nn_type == "mlp":
        xtrain, ytrain, xval, yval, xtest, ytest, class_weights = preprocess_data_mlp(
            xtrain_raw, xtest_raw, ytrain_raw, ytest_raw
        )
    elif args.nn_type == "cnn":
        xtrain, ytrain, xval, yval, xtest, ytest, class_weights = preprocess_data_cnn(
            xtrain_raw, xtest_raw, ytrain_raw, ytest_raw
        )
        # —————————————————————————————————————————————————————————————
        # sanity‐checks:
        print("Train images shape:", xtrain.shape)  # (N, C, H, W) for CNN
        print("Validation images shape:", xval.shape)
        print("Test images shape: ", xtest.shape)

        print("Class counts:", np.bincount(ytrain.astype(int)))
        print("Class weights:", class_weights)
        # —————————————————————————————————————————————————————————————

    else:
        raise ValueError("Unknown --nn_type. Use 'mlp' or 'cnn'")

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if args.test:
        # Concatenate training + validation
        train_data = np.concatenate([xtrain, xval], axis=0)
        train_labels = np.concatenate([ytrain, yval], axis=0)
        eval_data = xtest
        eval_labels = ytest
        set_type = "Test set"
    else:
        train_data = xtrain
        train_labels = ytrain
        eval_data = xval
        eval_labels = yval
        set_type = "Validation set"

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = len(np.unique(ytrain))
    if args.nn_type == "mlp":
        input_size = xtrain.shape[1]  # should be 28*28*3 = 2352
        model = MLP(input_size=input_size, n_classes=n_classes)
    elif args.nn_type == "cnn":
        input_channels = xtrain.shape[1]  # After transpose: should be 3 (RGB)
        model = CNN(input_channels=input_channels, n_classes=n_classes)

    summary(model)

    # Trainer object
    method_obj = Trainer(
        model=model,
        lr=args.lr,
        epochs=args.max_iters,
        batch_size=args.nn_batch_size,
        class_weights=class_weights
    )

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(train_data, train_labels)

    # Predict on unseen data
    preds_eval = method_obj.predict(eval_data)

    ## Report results: performance on train and valid/test sets
    acc_train = accuracy_fn(preds_train, train_labels)
    f1_train = macrof1_fn(preds_train, train_labels)
    print(f"\nTrain set: accuracy = {acc_train:.2f}% - F1-score = {f1_train:.4f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc_eval = accuracy_fn(preds_eval, eval_labels)
    f1_eval = macrof1_fn(preds_eval, eval_labels)
    print(f"{set_type}: accuracy = {acc_eval:.2f}% - F1-score = {f1_eval:.4f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")

    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (try 1e-3 or 1e-4)")
    parser.add_argument('--max_iters', type=int, default=100, help="Number of epochs (try 30–50 for CNN)")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)