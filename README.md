# DA6401 - Assignment 1

This repository contains the code files for **Assignment 1** of the course **DA6401**. The project involves training a neural network using various optimization techniques and hyperparameter tuning.

## Contents

### **Core Scripts**
- **`main.py`**: The primary training script, which implements the neural network training based on the assignment's specifications. Run it using:
  ```sh
  python main.py [optional arguments]
  ```
  or
  ```sh
  python3 main.py [optional arguments]
  ```
  where `[optional arguments]` are the command-line specifications for customizing training.

- **`train.py`**: Used for logging the **confusion matrix**, which helps in evaluating the model's classification performance.

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
- **Confusion Matrix Logging**: Execute `train.py` to generate and log confusion matrices for evaluating model predictions.
- **Hyperparameter Tuning**: Use `sweep_train.py` along with the provided sweep configuration files to explore different model settings.

## Wandb Report
The assignment report prepared on the wandb platform can be found: https://wandb.ai/ee20d201-indian-institute-of-technology-madras/DA6401_Assignment_1/reports/DA-6401-Assignment-1--VmlldzoxMTY4ODcwMA?accessToken=k8wh6e8bls5jz4p52w4t6rdzgsmby7ry6n18pd2qxeid893p3z2lazq6ok1ojru8. 

## Additional Information
For further details on how to modify hyperparameters or customize training, refer to the **code comments** inside the scripts.

---

This repository provides all necessary scripts and configurations to complete **Assignment 1** efficiently. Happy coding!

