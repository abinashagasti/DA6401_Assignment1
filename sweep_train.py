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
        
    wandb.init() # Initialize wandb run
    # wandb.init(resume="allow")
    config = wandb.config # Config for wandb sweep

    # Experiment name
    wandb.run.name = f"Experiment-hl_{config.num_layers}_ls_{config.hidden_size}_lr_{config.eta:.5f}_ \
               bs_{config.batch_size}_ac_{config.activation}_op_{config.optimizer}_ep_{config.epochs}_wi_{config.weight_init}_wd_{config.weight_decay}"
    
    # wandb.run.name = f"Experiment-hl_3_ls_128_lr_1e-5_ \
    #            bs_128_ac_relu_op_adam_ep_50_wi_xavier_wd_0.0005"


    # Initialize neural network object and train the model using hyperparameters from config
    nn = neural_network(input_size=784, num_layers=config.num_layers, output_size=10, hidden_layer_size=config.hidden_size, eta=config.eta, \
                        activation_method=config.activation, weight_initialization_method=config.weight_init, loss="cross_entropy")
    nn.gradient_descent(x_train, y_train, batch_size=config.batch_size, max_epochs=config.epochs, optimizer=config.optimizer, \
                        weight_decay=config.weight_decay, patience=3, learning_rate_annealing=True, wandb_log=True)

    # nn = neural_network(input_size=784, num_layers=3, output_size=10, hidden_layer_size=128, eta=1e-5, \
    #                     activation_method="relu", weight_initialization_method='xavier', loss="cross_entropy")
    # nn.gradient_descent(x_train, y_train, batch_size=128, max_epochs=50, optimizer="adam", \
    #                     weight_decay=0.0005, patience=3, learning_rate_annealing=True, wandb_log=True)
    
    wandb.finish() # End wandb run

with open("sweep_config_4.yaml", "r") as file:
    sweep_config = yaml.safe_load(file) # Read yaml file to store hyperparameters 

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_1")
entity = "ee20d201-indian-institute-of-technology-madras"
project = "DA6401_Assignment_1"
# api = wandb.Api()
# sweep_id = "w4t27t1b"
# sweep_id = "smvb01mt"
# sweep_id = "31sm4wfx"
# sweep_id = "dwruh8o9"

# sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

# if len(sweep.runs) >= 10:
#     api.stop_sweep(sweep.id)
#     print(f"Sweep {sweep.id} stopped after {len(sweep.runs)} runs.")

# Start sweep agent
wandb.agent(sweep_id, function=train, count=10, project=project)  # Run 10 experiments

