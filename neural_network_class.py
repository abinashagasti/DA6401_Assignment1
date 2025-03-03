import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from tqdm import tqdm

class neural_network:
    def __init__(self, input_size=784, num_layers=3, output_size=10, hidden_layer_size=64, eta=0.001, activation_method="sigmoid", weight_initialization_method="random"):
        self.input_size = input_size
        self.num_layers = num_layers # L-1
        self.output_size = output_size
        self.eta = eta # learning rate
        self.hidden_layer_size = np.ones((num_layers,), dtype=int)*hidden_layer_size
        self.layer_size = np.hstack(([input_size],self.hidden_layer_size,[output_size]))
        self.weight_initialization_method = weight_initialization_method
        self.initialize_parameters()
        self.reset_gradients()
        self.activation_method = activation_method

    def initialize_parameters(self):
        if self.weight_initialization_method=="random":
            self.weights = {f"W{i+1}": np.random.randn(self.layer_size[i+1], self.layer_size[i]) * 0.01 for i in range(self.num_layers + 1)}
            self.biases = {f"b{i+1}": np.random.randn(self.layer_size[i+1],1) * 0.01 for i in range(self.num_layers + 1)}
            self.a = {f"a{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}        
            self.h = {f"h{i+1}": np.zeros((self.layer_size[i+1],1)) for i in range(self.num_layers + 1)}
        elif self.weight_initialization_method=="xavier":
            std = np.sqrt(1.0 / self.input_size)
            return np.random.randn(self.output_size, self.input_size) * std

    def reset_gradients(self):
        self.gradients = {f"a{i+1}": np.zeros(self.a[f"a{i+1}"].shape) for i in range(self.num_layers + 1)}
        self.gradients.update({f"h{i+1}": np.zeros(self.h[f"h{i+1}"].shape) for i in range(self.num_layers + 1)})
        self.gradients.update({f"W{i+1}": np.zeros(self.weights[f"W{i+1}"].shape) for i in range(self.num_layers + 1)})
        self.gradients.update({f"b{i+1}": np.zeros(self.biases[f"b{i+1}"].shape) for i in range(self.num_layers + 1)})

    def activation(self, a):
        if self.activation_method=="sigmoid":
            threshold = 700
            return np.where(a > threshold, 1, np.where(a < -threshold, 0, 1 / (1 + np.exp(-np.clip(a, -threshold, threshold)))))
        elif self.activation_method=="relu":
            return np.maximum(0, a)
        elif self.activation_method=="tanh":
            return np.tanh(a)
        else:
            raise Exception("Incorrect activation function input!")
        
    def activation_derivative(self, a):
        if self.activation_method=="sigmoid":
            return self.activation(a)*(1-self.activation(a)) # works only for sigmoid
        elif self.activation_method=="relu":
            return np.where(a > 0, 1, 0)
        elif self.activation_method=="tanh":
            return 1-np.tanh(a)**2
        else:
            raise Exception("Incorrect activation function input!")

    def output(self, a):
        exp_shifted = np.exp(a - np.max(a))  # Prevent overflow
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)
        # return np.exp(a)/(np.sum(np.exp(a))+1e-9)

    def forward_propagation(self, input):
        self.a["a1"] = np.matmul(self.weights["W1"], input) + self.biases["b1"]
        for i in range(1,self.num_layers + 1):
            self.h[f"h{i}"] = self.activation(self.a[f"a{i}"])
            self.a[f"a{i+1}"] = np.matmul(self.weights[f"W{i+1}"], self.h[f"h{i}"]) + self.biases[f"b{i+1}"]
        self.h[f"h{self.num_layers+1}"] = self.output(self.a[f"a{self.num_layers+1}"])
        return self.h[f"h{self.num_layers+1}"]

    def one_hot_vector(self, output):
        e = np.zeros((self.output_size,1))
        e[output] = 1
        return e
    
    def backward_propagation(self, input, output):
        self.gradients[f"a{self.num_layers+1}"] = - (self.one_hot_vector(output)-self.h[f"h{self.num_layers+1}"])
        for k in range(self.num_layers+1,1,-1):
            self.gradients[f"W{k}"] = np.matmul(self.gradients[f"a{k}"], np.transpose(self.h[f"h{k-1}"]))
            self.gradients[f"b{k}"] = self.gradients[f"a{k}"]
            self.gradients[f"h{k-1}"] = np.matmul(np.transpose(self.weights[f"W{k}"]), self.gradients[f"a{k}"])
            self.gradients[f"a{k-1}"] = np.multiply(self.gradients[f"h{k-1}"], self.activation_derivative(self.a[f"a{k-1}"]))
        self.gradients["W1"] = np.matmul(self.gradients[f"a{1}"], np.transpose(input))
        self.gradients["b1"] = self.gradients["a1"]

    def loss(self, x_train, y_train):
        loss = 0
        for i in range(x_train.shape[0]):
            prediction = self.forward_propagation(x_train[i].reshape(self.input_size,1))
            epsilon = 1e-9
            prob = np.clip(prediction[y_train[i], 0], epsilon, 1 - epsilon)
            loss += -np.log(prob)
        return loss
    
    # def gradient_descent_old(self, x_train, y_train, batch_size=128):
    #     max_epochs = 1000
    #     batch_size = batch_size
    #     for epoch in tqdm(range(1,max_epochs+1), desc="Training Progress"): 
    #         loss = self.loss(x_train, y_train)/x_train.shape[0]
    #         print(f"Epoch {epoch}/{max_epochs} - Loss: {loss:.4f}")
    #         d_theta = {f"a{i+1}": np.zeros(self.a[f"a{i+1}"].shape) for i in range(self.num_layers + 1)}
    #         d_theta.update({f"h{i+1}": np.zeros(self.h[f"h{i+1}"].shape) for i in range(self.num_layers + 1)})
    #         d_theta.update({f"W{i+1}": np.zeros(self.weights[f"W{i+1}"].shape) for i in range(self.num_layers + 1)})
    #         d_theta.update({f"b{i+1}": np.zeros(self.biases[f"b{i+1}"].shape) for i in range(self.num_layers + 1)})
    #         for i in range(x_train.shape[0]):
    #             input = x_train[i].reshape(self.input_size,1)
    #             output = y_train[i]
    #             self.forward_propagation(input)
    #             self.backward_propagation(input, output)
    #             d_theta = {key: value + self.gradients[key] for key, value in d_theta.items()}
    #         print(f"Gradient Norm: {np.linalg.norm(d_theta['W1'])}")
    #         self.weights = {key: value - self.eta*d_theta[key] for key, value in self.weights.items()}
    #         self.biases = {key: value - self.eta*d_theta[key] for key, value in self.biases.items()}

    def train_validation_split(self, x_train, y_train, validation_ratio):
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

    def gradient_descent(self, x_train, y_train, batch_size=128, max_epochs=50):
        max_epochs = max_epochs
        validation_ratio = 0.1
        x_train_split, y_train_split, x_val, y_val = self.train_validation_split(x_train, y_train, validation_ratio)
        num_samples = x_train_split.shape[0]

        for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"): 
            # Sampling randomly
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train_split[indices]
            y_train_shuffled = y_train_split[indices]

            total_loss = 0

            for i in range(0, num_samples, batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]/255.0
                y_batch = y_train_shuffled[i:i + batch_size]

                # Initialize gradient changes (dw and db)
                d_theta = {f"W{i+1}": np.zeros(self.weights[f"W{i+1}"].shape) for i in range(self.num_layers + 1)}
                d_theta.update({f"b{i+1}": np.zeros(self.biases[f"b{i+1}"].shape) for i in range(self.num_layers + 1)})

                batch_loss = 0

                for j in range(len(x_batch)):
                    input = x_batch[j].reshape(self.input_size, 1)
                    output = y_batch[j]
                    
                    self.forward_propagation(input)
                    self.backward_propagation(input, output)
                    
                    d_theta = {key: value + self.gradients[key] for key, value in d_theta.items()}
                    batch_loss += -np.log(self.h[f"h{self.num_layers+1}"][output, 0] + 1e-9)  # Avoid log(0)

                # Average gradients over batch
                # d_theta = {key: value / len(x_batch) for key, value in d_theta.items()}

                # Update weights and biases
                self.weights = {key: value - self.eta * d_theta[key] for key, value in self.weights.items()}
                self.biases = {key: value - self.eta * d_theta[key] for key, value in self.biases.items()}

                total_loss += batch_loss

            # grad_norms_min = min(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # grad_norms_max = max(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # print(f"Minimum of max norm of each gradient: {grad_norms_min}")
            # print(f"Maximum of max norm of each gradient: {grad_norms_max}")

            avg_loss = total_loss / num_samples  # Average loss across all batches

            accurate_predictions = 0
            for i in range(x_val.shape[0]):
                y_pred = np.argmax(self.forward_propagation(x_val[i].reshape(self.input_size,1)))
                if y_pred==y_val[i]:
                    accurate_predictions += 1

            print(f"Epoch {epoch}/{max_epochs} - Loss: {avg_loss:.4f}, Validation Accuracy: {(accurate_predictions*1000)/num_samples: .4f}%")

