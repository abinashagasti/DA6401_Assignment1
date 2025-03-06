import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from tqdm import tqdm
import wandb
import yaml
import argparse

from utils import *
from neural_network_class import neural_network
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plot_sample_images(x_train, y_train)
nn = neural_network(num_layers=5, hidden_layer_size=128, eta=1e-4, activation_method="relu", weight_initialization_method="he", loss="cross_entropy")
nn.gradient_descent(x_train, y_train, batch_size=128, max_epochs=10, optimizer="adam", weight_decay=5e-4, patience=2, learning_rate_annealing=True, wandb_log=False)