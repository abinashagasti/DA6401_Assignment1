import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import yaml
import argparse

from utils import *
from neural_network_class import neural_network
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plot_sample_images(x_train, y_train, wandb_log=False)

# wandb.init(entity = "ee20d201-indian-institute-of-technology-madras",
#            project = "DA6401_Assignment_1",
#            name = "squared_error_2")

nn = neural_network(num_layers=3, hidden_layer_size=128, eta=1e-4, activation_method="relu", weight_initialization_method="xavier", loss="cross_entropy")
# nn.gradient_descent(x_train, y_train, batch_size=128, max_epochs=20, optimizer="adam", weight_decay=0.0005, patience=3, learning_rate_annealing=True, wandb_log=False)

nn.weights = np.load("best_weights_config2_ce.npy", allow_pickle=True).item()
nn.biases = np.load("best_biases_config2_ce.npy", allow_pickle=True).item()

y_pred, accurate_preds = return_predicted_output(nn, x_test, y_test)
print(f"Test accuracy: {(accurate_preds/y_test.shape[0])*100:.4f}")

# wandb.finish()

