import numpy as np
import matplotlib.pyplot as plt
from neural_network_class import neural_network
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from keras.datasets import fashion_mnist, mnist

def data_download(dataset):
    """
    Downloads data and return test and trains data samples

    Parameters:
    ----------
        dataset (str): Dataset name. fashion_mnist or mnist.

    Returns:
    -------
        numpy.ndarray : Train features
        numpy.ndarray : Train Labels
        numpy.ndarray : Test features
        numpy.ndarray : Test Labels
    """

    if dataset == "fashion_mnist":
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    elif dataset == "mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    return X_train, Y_train, X_test, Y_test

def plot_sample_images(x, y, wandb_run=None):
    """"
    Plot sample images from each class.

    Parameters:
    -----------
        x: Input data
        y: Output data
        wandb_run: Optional input to log on wandb api
    """
    indices = [np.random.choice(np.where(y == c)[0]) for c in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5)) 
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        axes[i].imshow(x[idx], cmap="gray")  # Display image
        axes[i].set_title(f"Class {i}")  # Set title
        axes[i].axis("off")  # Hide axes

    plt.tight_layout()
    if wandb_run is not None:
        wandb_run.log({"Sample Images": wandb.Image(fig)})
    plt.close(fig)

def create_confusion_matrix(y_actual, y_pred, wandb_run=None):
    """
    Creates confusion matrix plot

    Parameters:
    ----------
        y_actual (numpy.ndarray): Actual output
        y_pred (numpy.ndarray): predicted output
        wandb_run: optional argument containing wandb to log data

    Returns:
    -------
        None
    """
    # conf_matrix = confusion_matrix(y_actual, y_pred)
    # conf_matrix_df = pd.DataFrame(conf_matrix,
    #               index = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    #               columns = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

    all_labels = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # conf_matrix = confusion_matrix(y_actual, y_pred)

    if wandb_run is not None:
        # Log confusion matrix data
        wandb.log({"Confusion_matrix_fashion_mnist" : wandb.plot.confusion_matrix(preds=y_pred, y_true=y_actual,class_names=all_labels)})

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title("Confusion Matrix")

        # # Log Confusion Matrix as an Image
        # wandb.log({"confusion_matrix": wandb.Image(plt)})

        # table = wandb.Table(columns=["Actual", "Predicted", "Count"])
        # for i in range(len(all_labels)):  # Loop through actual classes
        #     for j in range(len(all_labels)):  # Loop through predicted classes
        #         table.add_data(all_labels[i], all_labels[j], conf_matrix[i, j])

        # # Log table to WandB
        # wandb.log({"Confusion Matrix Table": table})

def return_predicted_output(nn, x_test, y_test):
    """
    Return predicted output and probabilities. 

    Args:
        nn: neural network object
        x_test: test image samples
        y_test: test output labels

    Outputs:
        y_preds: Class predictions for each input test sample 
        accurate_preds: # Correct predictions in test set
        y_probs: Predicted probabilities of outputs 
    """
    y_preds = np.zeros(y_test.shape[0])
    y_probs = np.zeros((y_test.shape[0],10))
    accurate_preds = 0
    for i in range(x_test.shape[0]):
        nn.forward_propagation(x_test[i].reshape(-1, 1) / 255.0)  # Forward pass
        y_probs[i] = nn.h[f"h{nn.num_layers+1}"].flatten()
        y_preds[i] = np.argmax(y_probs[i])
        if y_preds[i]==y_test[i]:
            accurate_preds += 1

    return y_preds, accurate_preds, y_probs

def find_hard_samples(y_true, y_probs, top_k=10):
    """
    Identify the hardest samples where the model is most confident but wrong.

    Args:
        y_true (array-like): True class labels (shape: [num_samples]).
        y_probs (array-like): Predicted probabilities (shape: [num_samples, num_classes]).
        x_samples (array-like): Input samples (e.g., images) corresponding to y_true.
        top_k (int): Number of hardest samples to return.

    Returns:
        List of (index, true_label, predicted_label, confidence) for hardest samples.
    """
    y_pred = np.argmax(y_probs, axis=1)  # Get predicted class
    confidences = np.max(y_probs, axis=1)  # Get confidence of prediction

    incorrect_mask = (y_pred != y_true)  # Find misclassified samples
    incorrect_confidences = confidences[incorrect_mask]  # Get confidence for wrong predictions
    incorrect_indices = np.where(incorrect_mask)[0]  # Get indices of misclassified samples

    # Sort by highest confidence in wrong prediction
    sorted_indices = incorrect_indices[np.argsort(-incorrect_confidences)][:top_k]

    hard_samples = [(idx, y_true[idx], y_pred[idx], confidences[idx]) for idx in sorted_indices]

    return hard_samples

def plot_hard_samples(hard_samples, x_samples):
    """
    Plot the hardest misclassified samples.

    Args:
        hard_samples (list): List of (index, true_label, predicted_label, confidence).
        x_samples (array-like): Input samples (e.g., images).

    Returns:
        None (displays a plot)
    """
    fig, axes = plt.subplots(1, len(hard_samples), figsize=(15, 5))

    for i, (idx, true_label, pred_label, conf) in enumerate(hard_samples):
        axes[i].imshow(x_samples[idx], cmap="gray")
        axes[i].set_title(f"T: {true_label}, P: {pred_label}\nConf: {conf:.2f}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def check_forward_prop(nn, x, y):
    # Check correctness of forward propagation algorithm manually for a 3 hidden layer neural network. 
    y_check = nn.output(nn.weights["W4"]@nn.activation(nn.weights["W3"]@nn.activation(nn.weights["W2"]@nn.activation(nn.weights["W1"]@x \
                                                                                                                 +nn.biases["b1"])+nn.biases["b2"])+nn.biases["b3"])+nn.biases["b4"])
    if np.all(y==y_check):
        print("Forward propagation works!")
    else:
        print("Mistake in forward propagation!")

def check_backward_prop(nn, y_class, y):
    # Check correctness of backward propagation algorihtm manually for a 3 hidden layer neural network. 
    a4 = np.all(nn.gradients["a4"]==y-nn.one_hot_vector(y_class))
    W4 = np.all(nn.gradients["W4"]==nn.gradients["a4"]@np.transpose(nn.h["h3"]))
    b4 = np.all(nn.gradients["b4"]==nn.gradients["a4"])
    h3 = np.all(nn.gradients["h3"]==np.transpose(nn.weights["W4"])@nn.gradients["a4"])
    a3 = np.all(nn.gradients["a3"]==np.multiply(nn.gradients["h3"], nn.activation_derivative(nn.a["a3"])))
    if a4 and W4 and b4 and h3 and a3:
        print("Backward propagation works!")
    else:
        print("Mistake in backward propagation!")

def run_forward_backward_prop(x_train, y_train):
    nn = neural_network(hidden_layer_size=[128,64,32])
    x = x_train[0].reshape(784,1)
    y = nn.forward_propagation(x)

    # Checks forward propagation code
    check_forward_prop(nn, x, y)

    nn.backward_propagation(x, y_train[0])
    # Checks backward propagation code
    check_backward_prop(nn, y_train[0], y)