# DA6401 - Assignment 1

This repository contains the code files for **Assignment 1** of the course **DA6401**. The project involves training a neural network using various optimization techniques and hyperparameter tuning.

**GitHub Link**: https://github.com/abinashagasti/DA6401_Assignment1

## Contents

### **Core Scripts**
- **`train.py`**: The primary training script, which implements the neural network training based on the assignment's specifications. Run it using:
  ```sh
  python train.py [optional arguments]
  ```
  or
  ```sh
  python3 train.py [optional arguments]
  ```
  where `[optional arguments]` are the command-line specifications for customizing training.

- **`main.py`**: Used for logging the **confusion matrix**, which helps in evaluating the model's classification performance.

- **`sweep_train.py`**: Designed to perform **sweep training**, which explores different hyperparameter configurations to find the best set of values.

- **`utils.py`**: A collection of helper functions that assist various parts of the training and evaluation process.

### **Hyperparameter Sweep Configurations**
The repository includes several YAML configuration files for hyperparameter sweeps:
- **`sweep_config.yaml`**
- **`sweep_config_2.yaml`**
- **`sweep_config_3.yaml`**
- **`sweep_config_4.yaml`**

These files define different sets of hyperparameters that were explored for **Question 4** of the assignment to optimize model performance.

## Logging and Monitoring
- **Confusion Matrix Logging**: Execute `main.py` to generate and log confusion matrices for evaluating model predictions.
- **Hyperparameter Tuning**: Use `sweep_train.py` along with the provided sweep configuration files to explore different model settings.

## Wandb Report
The assignment report prepared on the wandb platform can be found: https://wandb.ai/ee20d201-indian-institute-of-technology-madras/DA6401_Assignment_1/reports/DA-6401-Assignment-1--VmlldzoxMTY4ODcwMA?accessToken=k8wh6e8bls5jz4p52w4t6rdzgsmby7ry6n18pd2qxeid893p3z2lazq6ok1ojru8. 

## Additional Information
For further details on how to modify hyperparameters or customize training, refer to the **code comments** inside the scripts.

"best_weights_config1_ce.npy" and "best_biases_config1_ce.npy" stores the best weights and biases obtained using the hyperparameter configuration:
epochs: 30
batch_size: 32
loss: "cross_entropy"
optimizer: "momentum"
learning_rate: 0.0025
momentum: 0.9
beta: 0.9
beta1: 0.9
beta2: 0.999
epsilon: 1e-9
weight_decay: 0.0005
weight_init: "Xavier"
num_layers: 3
hidden_size: 128
activation: "ReLu"

The above happens to be the best model out of all the sweeps performed on wandb. 

"best_weights_config2_ce.npy" and "best_biases_config2_ce.npy" stores the best weights and biases obtained using the hyperparameter configuration:
epochs: 30
batch_size: 128
loss: "cross_entropy"
optimizer: "adam"
learning_rate: 1e-4
momentum: 0.9
beta: 0.9
beta1: 0.9
beta2: 0.999
epsilon: 1e-9
weight_decay: 0.0005
weight_init: "Xavier"
num_layers: 3
hidden_size: 128
activation: "ReLu"

Another set of parameters is stored using the "squared_error" loss function. The names have been appropriately modified to contain "xx...xx_se.npy" instead of "xx...xx_ce.npy". 
---

