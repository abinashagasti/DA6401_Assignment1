import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from tqdm import tqdm

from utils import *
from neural_network_class import neural_network
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# plot_sample_images(x_test, y_test)

# run_forward_backward_prop(x_train, y_train)

nn = neural_network(num_layers=3,hidden_layer_size=128,eta=1e-3,activation_method="sigmoid")
nn.gradient_descent(x_train, y_train, batch_size=64, max_epochs=5, optimizer="adam", weight_decay=0)