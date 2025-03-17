import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from tqdm import tqdm
import wandb

class neural_network:
    def __init__(self, input_size=784, num_layers=3, output_size=10, hidden_layer_size=64, eta=0.001, activation_method="ReLu", weight_initialization_method="Xavier", loss="cross_entropy", epsilon=1e-9):
        """
        Initializes a neural network with the specified architecture and parameters.

        Parameters:
        -----------
        input_size : int, default=784
            Number of input neurons.
        
        num_layers : int, default=3
            Number of hidden layers in the network.

        output_size : int, default=10
            Number of output neurons.
        
        hidden_layer_size : int, default=64
            Number of neurons in each hidden layer (same size for all hidden layers).

        eta : float, default=0.001
            Learning rate for gradient descent optimization.

        activation_method : str, default="ReLu"
            Activation function used in the hidden layers (choices: "sigmoid", "ReLu", "tanh").

        weight_initialization_method : str, default="Xavier"
            Method for weight initialization (choices: "random", "Xavier", "He").

        loss : str, default="cross_entropy"
            Loss function to use for training (choices: "cross_entropy", "mse").

        epsilon : float, default=1e-9
            A small constant to avoid division by zero in numerical computations.

        Returns:
        --------
        None
        """
        self.input_size = input_size # sets input size
        self.num_layers = num_layers # sets number of hidden layers L-1
        self.output_size = output_size # sets output layer size
        self.eta = eta # sets learning rate
        # sets size of all neural network layers, hidden layers have same size
        self.hidden_layer_size = np.ones((num_layers,), dtype=int)*hidden_layer_size 
        self.layer_size = np.hstack(([input_size],self.hidden_layer_size,[output_size]))
        self.weight_initialization_method = weight_initialization_method # sets weight initialization method
        self.initialize_parameters() # initialization of weights and biases based on the weight initialization method
        self.reset_gradients() # resets all gradients to zero
        self.activation_method = activation_method # sets to activation function used in neural network
        self.loss = loss # loss function used in output layer
        self.epsilon = epsilon # epsilon value used for optimization algorithms (to avoid numerical overflow)

    def initialize_parameters(self):
        """
        Initializes the weights, biases, activations, and hidden layer outputs for the neural network
        based on the selected weight initialization method.

        Weight Initialization Methods:
        --------------------------------
        1. "random"  -> Small random values drawn from a normal distribution (default scaling: 0.01)
        2. "xavier"  -> Xavier (Glorot) initialization: Uniform distribution with range sqrt(6 / (fan_in + fan_out))
        3. "he"      -> He initialization: Normal distribution scaled by sqrt(2 / fan_in)

        Attributes Initialized:
        ------------------------
        - self.weights: Dictionary storing weight matrices for each layer
        - self.biases: Dictionary storing bias vectors for each layer
        - self.a: Dictionary storing pre-activation values for each layer (z = Wx + b)
        - self.h: Dictionary storing post-activation values for each layer (h = activation(z))

        Returns:
        --------
        None
        """
        if self.weight_initialization_method=="random":
            # Initializes weights and biases to small random values
            self.weights = {f"W{i+1}": np.random.randn(self.layer_size[i+1], self.layer_size[i]) * 0.01 for i in range(self.num_layers + 1)}
            self.biases = {f"b{i+1}": np.random.randn(self.layer_size[i+1],1) * 0.01 for i in range(self.num_layers + 1)}
            # Initializes preactivation and activation values to zero for each layer
            self.a = {f"a{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}        
            self.h = {f"h{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}
        elif self.weight_initialization_method=="xavier" or self.weight_initialization_method=="Xavier":
            # Initializes weights and biases using xavier initialization
            # Typically weights are sampled from a uniform distribution within a certain range 
            # defined by the number of neurons in input and output layer
            self.weights = {f"W{i+1}": np.random.uniform(-np.sqrt(6/(self.layer_size[i+1]+self.layer_size[i])) , np.sqrt(6/(self.layer_size[i+1]+self.layer_size[i])),\
                                                          (self.layer_size[i+1], self.layer_size[i])) for i in range(self.num_layers + 1)}
            # self.biases = {f"b{i+1}": np.random.uniform(-np.sqrt(6/(1+self.layer_size[i+1])), np.sqrt(6/(1+self.layer_size[i+1])), \
                                                        # (self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}
            # Biases are initialized randomly to small values even in xavier initialization 
            self.biases = {f"b{i+1}": np.random.randn(self.layer_size[i+1],1) * 0.01 for i in range(self.num_layers + 1)}
            # Initializes preactivation and activation values to zero for each layer
            self.a = {f"a{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}        
            self.h = {f"h{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}
        elif self.weight_initialization_method=="he" or self.weight_initialization_method=="He":
            # Initializes weights and biases using He initialization
            # Weights are drawn from a normal distribution based on the input layer size
            self.weights = {f"W{i+1}": np.random.randn(self.layer_size[i+1], self.layer_size[i]) * np.sqrt(2 / self.layer_size[i]) for i in range(self.num_layers + 1)}
            # Biases are again initialized randomly to small values
            self.biases = {f"b{i+1}": np.random.randn(self.layer_size[i+1],1) * 0.01 for i in range(self.num_layers + 1)}
            # Initializes preactivation and activation values to zero for each layer
            self.a = {f"a{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}        
            self.h = {f"h{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}

    def reset_gradients(self):
        """
        Resets the gradients of weights, biases, activations, and hidden layer outputs to zero.

        Purpose:
        --------
        - This function ensures that gradients from the previous training iteration do not accumulate.
        - It initializes all gradients as zero arrays of the same shape as their corresponding parameters.

        Gradients Reset:
        ----------------
        - "a{i+1}" → Gradients of pre-activation values
        - "h{i+1}" → Gradients of post-activation values 
        - "W{i+1}" → Gradients of weight matrices
        - "b{i+1}" → Gradients of bias vectors

        Returns:
        --------
        - self.gradients: Dictionary containing zeroed gradients for all layers.
        """
        self.gradients = {f"a{i+1}": np.zeros(self.a[f"a{i+1}"].shape) for i in range(self.num_layers + 1)}
        self.gradients.update({f"h{i+1}": np.zeros(self.h[f"h{i+1}"].shape) for i in range(self.num_layers + 1)})
        self.gradients.update({f"W{i+1}": np.zeros(self.weights[f"W{i+1}"].shape) for i in range(self.num_layers + 1)})
        self.gradients.update({f"b{i+1}": np.zeros(self.biases[f"b{i+1}"].shape) for i in range(self.num_layers + 1)})

        return self.gradients

    def activation(self, a):
        """
        Applies the selected activation function element-wise to the input array 'a'.

        Purpose:
        --------
        - This function defines different activation functions that introduce non-linearity into the neural network.
        - The activation function choice is specified by `self.activation_method`.

        Supported Activation Functions:
        --------------------------------
        - "sigmoid"  : Sigmoid function, squashes input between 0 and 1.
        - "relu" / "ReLu" : Rectified Linear Unit (ReLU), outputs max(0, a).
        - "tanh" : Hyperbolic tangent function, squashes input between -1 and 1.
        - "identity" : Identity function, outputs input as it is.

        Handling Extreme Values:
        ------------------------
        - For "sigmoid", large positive values can cause overflow errors in `np.exp(-a)`, leading to numerical instability.
        - To prevent this, values are clipped between -700 and 700 using `np.clip(a, -threshold, threshold)`.
        - Large positive values are approximated to 1, and large negative values are approximated to 0.

        Parameters:
        -----------
        a : np.ndarray
            The input array to apply activation function on.

        Returns:
        --------
        np.ndarray
            The activated output of the same shape as input.

        Raises:
        -------
        Exception : If an unsupported activation function is provided.
        """
        if self.activation_method=="sigmoid":
            threshold = 700 # Prevents numerical instability in exponential calculations
            return np.where(a > threshold,1, np.where(a < -threshold, 0, 1 / (1 + np.exp(-np.clip(a, -threshold, threshold)))))
            # If input is too large, approximate sigmoid output to 1. If input is too small, approximate sigmoid output to 0. 
            # Compute sigmoid normally for stable values.
        elif self.activation_method in ["relu","ReLu"]:
            return np.maximum(0, a)
            # ReLU: If a > 0, return a; else return 0
        elif self.activation_method=="tanh":
            return np.tanh(a)
            # Tanh: Squashes values between -1 and 1
        elif self.activation_method=="identity":
            return a
            # Identity function simply returns the input
        else:
            raise Exception("Incorrect activation function input!")
            # Handle invalid activation function choices
        
    def activation_derivative(self, a):
        """
        Computes the derivative of the activation function with respect to the input 'a'.

        Purpose:
        --------
        - This function calculates the gradient of the selected activation function.
        - Used during backpropagation to update weights in a neural network.

        Supported Activation Functions and Their Derivatives:
        -----------------------------------------------------
        - "sigmoid"  : Derivative is sigmoid(a) * (1 - sigmoid(a)).
        - "relu" / "ReLu" : Derivative is 1 for positive inputs, 0 for negative inputs.
        - "tanh" : Derivative is 1 - tanh²(a).
        - "identity" : Derivative is always 1.

        Parameters:
        -----------
        a : np.ndarray
            The input array where the derivative is evaluated.

        Returns:
        --------
        np.ndarray
            The derivative of the activation function evaluated at 'a'.

        Raises:
        -------
        Exception : If an unsupported activation function is provided.
        """
        if self.activation_method=="sigmoid":
            # Derivative of sigmoid: sigmoid(a) * (1 - sigmoid(a))
            return self.activation(a)*(1-self.activation(a))
        elif self.activation_method in ["relu", "ReLu"]:
            # Derivative of ReLU: 1 if a > 0, else 0
            return np.where(a > 0, 1, 0)
        elif self.activation_method=="tanh":
            # Derivative of tanh: 1 - tanh^2(a)
            return 1-np.tanh(a)**2
        elif self.activation_method=="identity":
            # Derivative of identity function is always 1
            return np.ones_like(a)
        else:
            raise Exception("Incorrect activation function input!")

    def output(self, a):
        """
        Computes the softmax activation function for the given input 'a'.
        
        Purpose:
        --------
        - Converts raw network outputs (logits) into probabilities.
        - Ensures that the sum of probabilities across all classes is 1.
        - Used in the output layer of a classification neural network.
        
        Parameters:
        -----------
        a : np.ndarray
            The input array (logits) representing raw scores from the final layer.

        Returns:
        --------
        np.ndarray
            The output probabilities after applying the softmax function.
        """
        # Compute the shifted exponentials to avoid numerical overflow.
        exp_shifted = np.exp(a - np.max(a))
        # Normalize by dividing each exponentiated value by the sum of all exponentiated values.
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

    def forward_propagation(self, input):
        """
        Performs forward propagation through the neural network.

        Purpose:
        --------
        - Computes activations for all layers, from input to output.
        - Applies the chosen activation function at each hidden layer.
        - Uses the softmax function in the final layer for classification.
        - Stores all these preactivation and activation values in corresponding class attributes. 

        Parameters:
        -----------
        input : np.ndarray
            The input vector (shape: input_size * 1) representing a single data point.

        Returns:
        --------
        np.ndarray
            The final output of the neural network after softmax activation.
        """
        self.a["a1"] = np.matmul(self.weights["W1"], input) + self.biases["b1"] # a1 = W1*x + b1
        for i in range(1,self.num_layers + 1):
            self.h[f"h{i}"] = self.activation(self.a[f"a{i}"]) # h(i) = g(a(i))
            self.a[f"a{i+1}"] = np.matmul(self.weights[f"W{i+1}"], self.h[f"h{i}"]) + self.biases[f"b{i+1}"] # a(i+1) = W(i+1)*h(i) + b(i+1)
        self.h[f"h{self.num_layers+1}"] = self.output(self.a[f"a{self.num_layers+1}"]) # h(L) = O(a(L))
        return self.h[f"h{self.num_layers+1}"] # returns predicted outputs

    def one_hot_vector(self, output):
        """
        Converts a class label into a one-hot encoded vector.

        Purpose:
        --------
        - Converts a scalar class label into a one-hot encoded vector of size (output_size * 1).
        - Used to convert true output into a probability distribution over the available classes.

        Parameters:
        -----------
        output : int
            The class label (an integer between 0 and output_size - 1).

        Returns:
        --------
        np.ndarray
            A one-hot encoded column vector of shape (output_size * 1).
        """
        e = np.zeros((self.output_size,1)) # Initialize zero vector of shape output_size * 1
        e[output] = 1 # Sets zero vector at true output index to 1 resulting in one hot vector. 
        return e
    
    def backward_propagation(self, input, output):
        """
        Performs backpropagation to compute gradients for weights and biases.

        Purpose:
        --------
        - Computes gradients of loss with respect to weights and biases.
        - Uses the chain rule to propagate errors backward through the network.
        - Supports both "cross_entropy" and "squared_error" loss functions.

        Parameters:
        -----------
        input : np.ndarray
            The input sample of shape (input_size, 1).
        output : int
            The true class label (integer between 0 and output_size - 1).

        Updates:
        --------
        - self.gradients: Dictionary storing gradients for weights, biases, and activations.
        """
        # Compute the gradient of the loss function with respect to the final layer activations
        if self.loss == "cross_entropy":
            # Gradient for cross-entropy loss: dL/dh = -(y_true - y_pred)
            self.gradients[f"a{self.num_layers+1}"] = - (self.one_hot_vector(output)-self.h[f"h{self.num_layers+1}"])
        elif self.loss == "squared_error" or self.loss == "mean_squared_error":
            # Gradient for squared error loss: dL/dh = 2 * (y_pred - y_true)
            self.gradients[f"a{self.num_layers+1}"] = 2 * (self.h[f"h{self.num_layers+1}"] - self.one_hot_vector(output))

        # Backpropagate the error through the network
        for k in range(self.num_layers+1,1,-1):
            # Compute weight gradient: dL/dW(k) = dL/da(k) * h(k-1)^T (previous layer activations)
            self.gradients[f"W{k}"] = np.matmul(self.gradients[f"a{k}"], np.transpose(self.h[f"h{k-1}"]))
            # Compute bias gradient: dL/db(k) = dL/da(k)
            self.gradients[f"b{k}"] = self.gradients[f"a{k}"]
            # Compute gradient for previous layer's activations: dL/dh_{k-1} = W(k)^T * dL/da(k)
            self.gradients[f"h{k-1}"] = np.matmul(np.transpose(self.weights[f"W{k}"]), self.gradients[f"a{k}"])
            # Compute gradient for previous layer's pre-activation values: dL/da_{k-1} = dL/dh_{k-1} * activation_derivative(a(k-1))
            self.gradients[f"a{k-1}"] = np.multiply(self.gradients[f"h{k-1}"], self.activation_derivative(self.a[f"a{k-1}"]))
        
        # Compute gradients for the first layer separately
        self.gradients["W1"] = np.matmul(self.gradients[f"a{1}"], np.transpose(input))
        self.gradients["b1"] = self.gradients["a1"]

    def loss(self, x_train, y_train):
        """
        Computes the total loss for the given training data.

        Purpose:
        --------
        - Evaluates the loss function over the entire training dataset.
        - Uses cross-entropy loss, assuming a classification problem with one-hot encoded labels.

        Parameters:
        -----------
        x_train : np.ndarray
            The training data, shape (num_samples, input_size).
        y_train : np.ndarray
            The true class labels, shape (num_samples,).

        Returns:
        --------
        loss : float
            The total loss over all training samples.
        """
        loss = 0
        for i in range(x_train.shape[0]):
            prediction = self.forward_propagation(x_train[i].reshape(self.input_size,1)) # obtain predicted output
            prob = np.clip(prediction[y_train[i], 0], self.epsilon, 1 - self.epsilon)  # Extract the predicted probability of the correct class (y_train[i])
            loss += -np.log(prob) # Add loss at datapoint i to total loss
        return loss

    def train_validation_split(self, x_train, y_train, validation_ratio):
        """
        Splits the dataset into training and validation sets.

        Purpose:
        --------
        - Randomly partitions the given training data into training and validation sets.
        - Helps evaluate model performance on unseen validation data.

        Parameters:
        -----------
        x_train : np.ndarray
            The input training data, shape (num_samples, input_size).
        y_train : np.ndarray
            The corresponding labels, shape (num_samples,).
        validation_ratio : float
            The proportion of samples to be used for validation (e.g., 0.2 for 20% validation).

        Returns:
        --------
        x_train_split : np.ndarray
            Training subset of the input data.
        y_train_split : np.ndarray
            Training subset of the labels.
        x_val : np.ndarray
            Validation subset of the input data.
        y_val : np.ndarray
            Validation subset of the labels.
        """
        num_samples = x_train.shape[0]
        num_val = int(num_samples * validation_ratio)  # Number of validation samples

        # Generate shuffled indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split indices into training and validation
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        # Create training and validation sets
        x_train_split = x_train[train_indices]
        y_train_split = y_train[train_indices]
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]

        return x_train_split, y_train_split, x_val, y_val
    
    def optimizer_step(self, optimizer, d_theta, momentums, velocities, beta_momentum, beta_rmsprop, beta1, beta2, t, weight_decay):
        """
        Updates the weights and biases using the specified optimization algorithm.

        Parameters:
        -----------
        optimizer : str
            The optimization method to use ("sgd", "momentum", "nag", "rmsprop", "adam", "nadam").
        d_theta : dict
            The accumulated gradients of the weights and biases over the mini-batch computed using backpropagation.
        momentums : dict
            Dictionary to store moving averages of gradients (used in momentum-based optimizers).
        velocities : dict
            Dictionary to store squared gradients (used in RMSProp, Adam and Nadam).
        beta_momentum : float
            Momentum coefficient (used in Momentum and NAG).
        beta_rmsprop : float
            Decay rate for squared gradients (used in RMSProp).
        beta1 : float
            First moment decay rate (used in Adam and Nadam).
        beta2 : float
            Second moment decay rate (used in Adam and Nadam).
        t : int
            Time step (used in Adam and Nadam for bias correction).
        weight_decay : float
            Regularization parameter for L2 weight decay.

        """

        # Stochastic Gradient Descent (SGD)
        if optimizer=="sgd":
            # Update weights with simple gradient descent and optional weight decay
            self.weights = {key: value - self.eta * (d_theta[key] + weight_decay*value) for key, value in self.weights.items()}
            # Update weights with simple gradient descent and optional weight decay
            self.biases = {key: value - self.eta * d_theta[key] for key, value in self.biases.items()}
        
        # Momentum-based Gradient Descent
        elif optimizer=="momentum":
            # Update momentum term
            momentums = {key: beta_momentum * value + d_theta[key] for key, value in momentums.items()}
            # Update weights using the momentum-adjusted gradients
            self.weights = {key: value - self.eta * (momentums[key] + weight_decay*value) for key, value in self.weights.items()}
            self.biases = {key: value - self.eta * momentums[key] for key, value in self.biases.items()}
        
        # Nesterov Accelerated Gradient (NAG)
        elif optimizer=="nag":
            # Compute intermediate momentum step
            momentums = {key: beta_momentum * value + d_theta[key] for key, value in momentums.items()}
            # Apply NAG update rule (uses a lookahead step)
            self.weights = {key: value - self.eta * (beta_momentum * momentums[key] + d_theta[key] + weight_decay*value) for key, value in self.weights.items()}
            self.biases = {key: value - self.eta * (beta_momentum * momentums[key] + d_theta[key]) for key, value in self.biases.items()}
        
        # RMSProp (Root Mean Square Propagation)
        elif optimizer=="rmsprop":
            # Compute exponentially weighted moving average of squared gradients
            momentums = {key: beta_rmsprop * value + (1-beta_rmsprop) * np.square(d_theta[key]) for key, value in momentums.items()}
            # Update weights and biases using RMSProp update rule
            self.weights = {key: value - (self.eta/ (np.sqrt(momentums[key]+self.epsilon))) * (d_theta[key] + weight_decay*value) for key, value in self.weights.items()}
            self.biases = {key: value - (self.eta/ (np.sqrt(momentums[key]+self.epsilon))) * d_theta[key] for key, value in self.biases.items()}
        
        # Adam
        elif optimizer=="adam":
            # Compute momentum (cumulative gradient) estimate (momentum)
            momentums = {key: beta1 * value + (1-beta1) * d_theta[key] for key, value in momentums.items()}
            # Compute velocities estimate (RMSProp-like, cumulative squared gradients)
            velocities = {key: beta2 * value + (1-beta2) * np.square(d_theta[key]) for key, value in velocities.items()}
            # Bias correction for moment estimates
            momentums_hat = {key: value / (1-pow(beta1,t)) for key, value in momentums.items()}
            velocities_hat = {key: value / (1-pow(beta2,t)) for key, value in velocities.items()}
            # Apply Adam update rule
            self.weights = {key: value * (1-weight_decay*self.eta) - self.eta/ (np.sqrt(velocities_hat[key])+self.epsilon) * (momentums_hat[key]) for key, value in self.weights.items()}
            self.biases = {key: value - self.eta/ (np.sqrt(velocities_hat[key])+self.epsilon) * momentums_hat[key] for key, value in self.biases.items()}

        # Nadam
        elif optimizer=="nadam":
            # Compute momentum (cumulative gradient) estimate (momentum)
            momentums = {key: beta1 * value + (1-beta1) * d_theta[key] for key, value in momentums.items()}
            # Bias corrected momentum values
            momentums_hat = {key: value / (1-pow(beta1,t)) for key, value in momentums.items()}
            # Compute velocities estimate (RMSProp-like, cumulative squared gradients)
            velocities = {key: beta2 * value + (1-beta2) * np.square(d_theta[key]) for key, value in velocities.items()}
            # Bias corrected velocity values
            velocities_hat = {key: value / (1-pow(beta2,t)) for key, value in velocities.items()}
            # Apply Nadam update rule
            self.weights = {key: value * (1-weight_decay*self.eta) - (self.eta/ (np.sqrt(velocities_hat[key])+self.epsilon)) * (beta1*momentums_hat[key] + \
                                                                                            ((1-beta1)/(1-pow(beta1,t)))*d_theta[key]) for key, value in self.weights.items()}
            self.biases = {key: value - (self.eta/ (np.sqrt(velocities_hat[key])+self.epsilon)) * (beta1*momentums_hat[key] + \
                                                                                            ((1-beta1)/(1-pow(beta1,t)))*d_theta[key]) for key, value in self.biases.items()}

    def gradient_descent(self, x_train, y_train, batch_size=128, max_epochs=50, optimizer="adam", weight_decay=0, patience=3, patience_stop=5,\
                         learning_rate_annealing=False, wandb_log=False, beta_momentum=0.9, beta_rmsprop=0.9, beta1=0.9, beta2=0.999):
        """
        Performs mini-batch gradient descent to optimize the neural network weights and biases.

        Parameters:
        - x_train (ndarray): Training input data.
        - y_train (ndarray): Corresponding training labels.
        - batch_size (int, optional): Number of samples per batch. Default is 128.
        - max_epochs (int, optional): Maximum number of training epochs. Default is 50.
        - optimizer (str, optional): Optimization algorithm ('sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'). Default is "adam".
        - weight_decay (float, optional): Regularization factor for weight decay. Default is 0 (no weight decay).
        - patience (int, optional): Number of epochs with no improvement before applying learning rate annealing. Default is 3.
        - patience_stop (int, optional): Number of epochs with no improvement before stopping training. Default is 5.
        - learning_rate_annealing (bool, optional): If True, reduces learning rate when validation loss stops improving. Default is False.
        - wandb_log (bool, optional): If True, logs training and validation metrics to Weights & Biases. Default is False.
        - beta_momentum (float, optional): Momentum factor for momentum-based optimizers. Default is 0.9.
        - beta_rmsprop (float, optional): Decay factor for RMSProp optimizer. Default is 0.9.
        - beta1 (float, optional): Exponential decay rate for the first moment estimate in Adam/Nadam. Default is 0.9.
        - beta2 (float, optional): Exponential decay rate for the second moment estimate in Adam/Nadam. Default is 0.999.

        Functionality:
        - Implements mini-batch gradient descent with various optimizers.
        - Uses a validation set for monitoring generalization and early stopping.
        - Supports learning rate annealing and weight decay.
        - Logs training progress with optional Weights & Biases integration.
        - Saves the best model parameters based on validation loss.

        Returns:
        - None (updates model parameters in-place).

        """
        validation_ratio = 0.1 # 10% of training set is kept as validation set
        # Split training data into training and validation set
        x_train_split, y_train_split, x_val, y_val = self.train_validation_split(x_train, y_train, validation_ratio) 
        num_samples = x_train_split.shape[0] # number of training samples

        # Initialize momentum and velocity dictionaries
        momentums = {key: np.zeros_like(value) for key, value in self.weights.items()}
        momentums.update({key: np.zeros_like(value) for key, value in self.biases.items()})
        velocities = momentums
        t = 0 # Set counter for adam and nadam updates

        best_validation_loss = np.inf # Initialize validation loss to +Inf
        patience_counter = 0 # Set a patience counter to 0

        for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"): 
            # Sampling randomly and shuffling training data
            indices = np.random.permutation(num_samples) 
            x_train_shuffled = x_train_split[indices]
            y_train_shuffled = y_train_split[indices]

            # Initialize total training loss and accurate predictions in training
            total_loss = 0 
            accurate_predictions_training = 0

            # Iterate through batches
            for i in range(0, num_samples, batch_size):
                # Obtain batch samples 
                x_batch = x_train_shuffled[i:i + batch_size]/255.0 
                y_batch = y_train_shuffled[i:i + batch_size]

                # Initialize accumulated gradients
                d_theta = {key: np.zeros_like(value) for key, value in self.weights.items()}
                d_theta.update({key: np.zeros_like(value) for key, value in self.biases.items()})

                batch_loss = 0 # Initialize batch loss to zero

                # Iterate within the batch through every data point
                for j in range(len(x_batch)):
                    input = x_batch[j].reshape(self.input_size, 1) 
                    output = y_batch[j]
                    
                    # Perform forward propagation and backward propagation for the current input and output
                    self.forward_propagation(input)
                    self.backward_propagation(input, output)

                    # Predicted output
                    y_pred = np.argmax(self.h[f"h{self.num_layers+1}"])
                    if y_pred == output:
                        accurate_predictions_training += 1 # Appropriately calculate accurate predictions while training
                    
                    # d_theta accumulates gradients across the batch
                    d_theta = {key: value + self.gradients[key] for key, value in d_theta.items()}
                    # Compute batch loss 
                    if self.loss == "cross_entropy":                     
                        batch_loss += -np.log(self.h[f"h{self.num_layers+1}"][output, 0] + self.epsilon)  # Avoid log(0)
                    elif self.loss == "squared_error" or self.loss == "mean_squared_error":
                        y_true = np.zeros((self.output_size,1))
                        y_true[output,0] = 1
                        batch_loss += np.sum(np.square(self.h[f"h{self.num_layers+1}"]-y_true))

                # Update weights and biases
                t += 1
                # Perform an optimization step
                self.optimizer_step(optimizer, d_theta, momentums, velocities, beta_momentum, beta_rmsprop, beta1, beta2, t, weight_decay)
                # Compute total loss over training data
                total_loss += batch_loss

            # grad_norms_min = min(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # grad_norms_max = max(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # print(f"Minimum of max norm of each gradient: {grad_norms_min}")
            # print(f"Maximum of max norm of each gradient: {grad_norms_max}")

            avg_training_loss = total_loss / num_samples  # Average loss across all batches

            # Initialize accurate predictions and loss for validation set
            accurate_predictions = 0
            validation_loss = 0
            # Iterate through all validation data
            for i in range(x_val.shape[0]):
                input = x_val[i].reshape(self.input_size,1) / 255.0
                output = y_val[i]

                self.forward_propagation(input) # Perform forward propagation
                validation_loss += -np.log(self.h[f"h{self.num_layers+1}"][output, 0] + self.epsilon) # Compute validation loss

                y_pred = np.argmax(self.h[f"h{self.num_layers+1}"]) # Predicted output
                if y_pred==output:
                    accurate_predictions += 1 # Compute accurate predictions in validation set

            validation_loss /= x_val.shape[0]
            training_accuracy = (accurate_predictions_training / num_samples) * 100 # Training accuracy
            validation_accuracy = (accurate_predictions / x_val.shape[0]) * 100 # Validation accuracy

            # Log data onto wandb if wandb_log==True
            if wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_training_loss,
                    "training_accuracy": training_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy
                })

            if validation_loss > best_validation_loss:
                # If current validation loss is higher than best validation loss then increase patience counter and patience stop counter.
                patience_counter += 1
                patience_stop_counter += 1 
                print(f"Epoch {epoch}/{max_epochs} - Training Loss: {avg_training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}%, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy: .4f}%")
                # Print training metrics. 
                if patience_counter >= patience:
                    # If patience counter is larger than patience threshold then perform learning rate annealing depending on patience counter.
                    # Also perform early stopping if patience stop counter is larger than patience stop threshold. 
                    if not learning_rate_annealing:
                        if patience_stop_counter>patience_stop:
                            print("Early stopping activated")
                            break
                    elif learning_rate_annealing:
                        if patience_stop_counter>patience_stop:
                            print("Early stopping activated")
                            break
                        print("Reloading best model and reducing learning rate by half ...")
                        self.weights = np.load("best_weights.npy", allow_pickle=True).item()
                        self.biases = np.load("best_biases.npy", allow_pickle=True).item()
                        self.eta /= 2
                        patience_counter = 0
                    
            else:
                # If validation loss is the best yet then reset best validation loss value and patience counters. Store current model parameters. 
                best_validation_loss = validation_loss
                patience_counter = 0
                patience_stop_counter = 0
                print(f"Epoch {epoch}/{max_epochs} - Training Loss: {avg_training_loss:.4f}, Training Accuracy: {(accurate_predictions_training / num_samples) * 100:.4f}%, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {(accurate_predictions / x_val.shape[0]) * 100: .4f}%")
                best_weights = self.weights
                best_biases = self.biases
                print(f"Storing current model parameters (epoch {epoch}) ...")
                # Store weights and biases if they provide the best validation loss upto this point. 
                np.save("best_weights.npy", best_weights)
                np.save("best_biases.npy", best_biases)

            