import numpy as np
import matplotlib.pyplot as plt
from neural_network_class import neural_network
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_sample_images(x, y, wandb_run=None):
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

    Returns:
    -------
        None
    """
    # conf_matrix = confusion_matrix(y_actual, y_pred)
    # conf_matrix_df = pd.DataFrame(conf_matrix,
    #               index = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    #               columns = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

    all_labels = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    conf_matrix = confusion_matrix(y_actual, y_pred)

    if wandb_run is not None:
        # wandb.log({"Confusion_matrix_fashion_mnist" : wandb.plot.confusion_matrix(preds=y_pred, y_true=y_actual,class_names=all_labels)})

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title("Confusion Matrix")

        # # Log Confusion Matrix as an Image
        # wandb.log({"confusion_matrix": wandb.Image(plt)})

        table = wandb.Table(columns=["Actual", "Predicted", "Count"])
        for i in range(len(all_labels)):  # Loop through actual classes
            for j in range(len(all_labels)):  # Loop through predicted classes
                table.add_data(all_labels[i], all_labels[j], conf_matrix[i, j])

        # Log table to WandB
        wandb.log({"Confusion Matrix Table": table})
        wandb.finish()

def return_predicted_output(nn, x_test, y_test):
    y_preds = np.zeros(y_test.shape[0])
    accurate_preds = 0
    for i in range(x_test.shape[0]):
        nn.forward_propagation(x_test[i].reshape(-1, 1) / 255.0)  # Forward pass
        y_preds[i] = np.argmax(nn.h[f"h{nn.num_layers+1}"])  # Get predicted class
        if y_preds[i]==y_test[i]:
            accurate_preds += 1

    return y_preds, accurate_preds

def log_hard_samples(y_true, y_pred_prob, x_samples):
    """Logs hardest misclassified samples with their confidence levels."""
    confidence = np.max(y_pred_prob, axis=1)  # Take max probability per sample
    misclassified = (y_true != np.argmax(y_pred_prob, axis=1))  # Identify errors

    # Select 10 hardest misclassified samples (lowest confidence)
    hardest_indices = np.argsort(confidence[misclassified])[:10]
    
    # table = wandb.Table(columns=["Image", "True Label", "Predicted Label", "Confidence"])
    # for idx in hardest_indices:
    #     table.add_data(wandb.Image(x_samples[idx]), y_true[idx], np.argmax(y_pred_prob[idx]), confidence[idx])

    # wandb.log({f"Hard Samples - {model_name}": table})

def check_forward_prop(nn, x, y):
    y_check = nn.output(nn.weights["W4"]@nn.activation(nn.weights["W3"]@nn.activation(nn.weights["W2"]@nn.activation(nn.weights["W1"]@x \
                                                                                                                 +nn.biases["b1"])+nn.biases["b2"])+nn.biases["b3"])+nn.biases["b4"])
    if np.all(y==y_check):
        print("Forward propagation works!")
    else:
        print("Mistake in forward propagation!")

def check_backward_prop(nn, y_class, y):
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