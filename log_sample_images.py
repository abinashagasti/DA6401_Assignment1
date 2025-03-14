from utils import *
import wandb
import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# wandb.init(entity = "ee20d201-indian-institute-of-technology-madras",
#            project = "DA6401_Assignment_1",
#            name = "plotting_sample_images")

wandb.init(entity = "ee20d201-indian-institute-of-technology-madras",
           project = "DA6401_Assignment_1",
           name = "confusion_matrix_table")

# plot_sample_images(x_train, y_train, wandb)

nn = neural_network(num_layers=3, hidden_layer_size=128, eta=1e-4, activation_method="relu", weight_initialization_method="xavier", loss="cross_entropy")
nn.gradient_descent(x_train, y_train, batch_size=128, max_epochs=20, optimizer="adam", weight_decay=5e-4, patience=3, learning_rate_annealing=True, wandb_log=False)

# nn.weights = np.load("best_weights.npy", allow_pickle=True).item()
# nn.biases = np.load("best_biases.npy", allow_pickle=True).item()

# Get Predictions
y_preds, accurate_preds = return_predicted_output(nn, x_test, y_test)

create_confusion_matrix(y_test, y_preds, wandb_run=wandb)

print(f"Test accuracy: {accurate_preds/y_test.shape[0]:.4f}")

# wandb.finish()
