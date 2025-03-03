import numpy as np
import matplotlib.pyplot as plt
from neural_network_class import neural_network

def plot_sample_images(x, y):
    indices = [np.random.choice(np.where(y == c)[0]) for c in range(10)]
    # print(indices)

    m = x.shape[0]

    fig, axes = plt.subplots(2, 5, figsize=(10, 5)) 
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        axes[i].imshow(x[idx], cmap="gray")  # Display image
        axes[i].set_title(f"Class {i}")  # Set title
        axes[i].axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()

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