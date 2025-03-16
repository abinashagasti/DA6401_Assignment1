import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import yaml
import argparse

from utils import *
from neural_network_class import neural_network


def main(args):
    """
    runs all the functions
    """
    user = args.wandb_entity
    project = args.wandb_project
    display_name = "test_run"
    if args.wandb_login:
        wandb.init(entity=user, project=project, name=display_name) # Initialize wandb experiment. 
        # config = wandb.config
        if args.dataset == "mnist":
            wandb.run.name = str(args.dataset) + "_lr_" + str(args.learning_rate) + "_opt_" + str(args.optimizer) + "_epoch_" + str(args.epochs) + "_bs_" + str(args.batch_size) + "_act_" + str(args.activation)
        else:
           wandb.run.name = "lr_" + str(args.learning_rate) + "_opt_" + str(args.optimizer) + "_epoch_" + str(args.epochs) + "_bs_" + str(args.batch_size) + "_act_" + str(args.activation)
        wandb_run = wandb
    else:
       wandb_run = None

    x_train, y_train, x_test, y_test = data_download(args.dataset) # Obtain dataset
    plot_sample_images(x_train,y_train,wandb_run) # Plot sample images
    # Initialize neural network object
    nn = neural_network(input_size=784, num_layers=args.num_layers, output_size=10, hidden_layer_size=args.hidden_size, eta=args.learning_rate,\
                        activation_method=args.activation, weight_initialization_method=args.weight_init, loss=args.loss, epsilon=args.epsilon)
    # Call gradient descent function
    nn.gradient_descent(x_train, y_train, batch_size=args.batch_size, max_epochs=args.epochs, optimizer=args.optimizer, weight_decay=args.weight_decay,\
                         learning_rate_annealing=args.lr_annealing, wandb_log=args.wandb_login, beta_momentum=args.momentum, beta_rmsprop=args.beta, beta1=args.beta1, beta2=args.beta2, patience_stop=10)

    # _, accurate_predictions, _ = return_predicted_output(nn, x_test, y_test)
    # print(f"Test accuracy: {(accurate_predictions/y_test.shape[0]) * 100:.4f}%")

    if args.wandb_login:
       wandb.finish()

if __name__=="__main__":
  # Set arguments for running current file. 
  # Use command python/python3 main.py followed by desired arguments. 
  parser = argparse.ArgumentParser()
  parser.add_argument("-wp","--wandb_project",default="DA6401_Assignment_1", help="Project name used to track experiments in Weights & Biases dashboard",type=str)
  parser.add_argument("-we","--wandb_entity",default="ee20d201-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard",type=str)
  parser.add_argument("-d","--dataset",default="fashion_mnist",choices = ["mnist", "fashion_mnist"], help="dataset to use for experiment",type=str)
  parser.add_argument("-e","--epochs",default=30,help="Number of epochs to train neural network",type=int)
  parser.add_argument("-b","--batch_size",default=32,help="Batch size used to train neural network",type=int)
  parser.add_argument("-l","--loss",default="cross_entropy",choices=["mean_squared_error", "cross_entropy"], help="Loss Function to train neural network",type=str)
  parser.add_argument("-o","--optimizer",default="momentum",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="optimizer to train neural network",type=str)
  parser.add_argument("-lr","--learning_rate",default=0.0025, help="Learning rate used to optimize model parameters",type=float)
  parser.add_argument("-m","--momentum",default=0.9, help="Momentum used by momentum and nag optimizers",type=float)
  parser.add_argument("-beta","--beta",default=0.9, help="Beta used by rmsprop optimizer",type=float)
  parser.add_argument("-beta1","--beta1",default=0.9, help="Beta1 used by adam and nadam optimizers",type=float)
  parser.add_argument("-beta2","--beta2",default=0.9, help="Beta2 used by adam and nadam optimizers",type=float)
  parser.add_argument("-eps","--epsilon",default=1e-9, help="Epsilon used by optimizers",type=float)
  parser.add_argument("-w_d","--weight_decay",default=0.0005, help="Weight decay used by optimizers",type=float)
  parser.add_argument("-w_i","--weight_init",default='Xavier',choices = ["random", "Xavier"], help="Weight initialization techniques",type=str)
  parser.add_argument("-nhl","--num_layers",default=3,help="Number of hidden layers used in feedforward neural network",type=int)
  parser.add_argument("-sz","--hidden_size",default=128,help="Number of hidden neurons in a feedforward layer",type=int)
  parser.add_argument("-a","--activation",default='ReLu',choices = ["identity", "sigmoid", "tanh", "ReLu"], help="Activation functions",type=str)
  parser.add_argument("-wbl","--wandb_login",default=True,action="store_false", help="Login data onto wandb.ai")
  parser.add_argument("-an","--lr_annealing",default=True,action="store_false",help="Learning rate annealing")
  args = parser.parse_args()
  main(args)
