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

def train():
        
    wandb.init() 
    config = wandb.config

    wandb.run.name = f"Experiment-hl_{config.num_layers}_ls_{config.hidden_size}_lr_{config.eta:.5f}_ \
               bs_{config.batch_size}_ac_{config.activation}_op_{config.optimizer}_ep_{config.epochs}_wi_{config.weight_init}_wd_{config.weight_decay}"

    nn = neural_network(input_size=784, num_layers=config.num_layers, output_size=10, hidden_layer_size=config.hidden_size, eta=config.eta, \
                        activation_method=config.activation, weight_initialization_method=config.weight_init, loss="cross_entropy")
    nn.gradient_descent(x_train, y_train, batch_size=config.batch_size, max_epochs=config.epochs, optimizer=config.optimizer, \
                        weight_decay=config.weight_decay, patience=3, learning_rate_annealing=True, wandb_log=True)
    
    wandb.finish()

with open("sweep_config.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="neural-network-sweep")

# Start the agent
wandb.agent(sweep_id, function=train, count=10)  # Run 10 experiments