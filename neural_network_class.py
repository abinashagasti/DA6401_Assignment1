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

        return self.gradients

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
    
    def return_gradients(self, weights, biases, x_batch, y_batch):
        a = {key: np.zeros_like(value) for key, value in self.a.items()}
        h = {key: np.zeros_like(value) for key, value in self.h.items()}
        gradients = self.gradients
        d_theta =  {key: np.zeros_like(value) for key, value in gradients.items()}

        for j in range(len(x_batch)):
            input = x_batch[j].reshape(self.input_size, 1)
            output = y_batch[j]

            a["a1"] = np.matmul(weights["W1"], input) + biases["b1"]
            for i in range(1,self.num_layers + 1):
                h[f"h{i}"] = self.activation(a[f"a{i}"])
                a[f"a{i+1}"] = np.matmul(weights[f"W{i+1}"], h[f"h{i}"]) + biases[f"b{i+1}"]
            h[f"h{self.num_layers+1}"] = self.output(a[f"a{self.num_layers+1}"])

            gradients[f"a{self.num_layers+1}"] = - (self.one_hot_vector(output)-h[f"h{self.num_layers+1}"])
            for k in range(self.num_layers+1,1,-1):
                gradients[f"W{k}"] = np.matmul(gradients[f"a{k}"], np.transpose(h[f"h{k-1}"]))
                gradients[f"b{k}"] = gradients[f"a{k}"]
                gradients[f"h{k-1}"] = np.matmul(np.transpose(weights[f"W{k}"]), gradients[f"a{k}"])
                gradients[f"a{k-1}"] = np.multiply(gradients[f"h{k-1}"], self.activation_derivative(a[f"a{k-1}"]))
            gradients["W1"] = np.matmul(gradients[f"a{1}"], np.transpose(input))
            gradients["b1"] = gradients["a1"]

            d_theta = {key: value + gradients[key] for key, value in d_theta.items()}

        return d_theta


    def gradient_descent(self, x_train, y_train, batch_size=128, max_epochs=50, optimizer="sgd"):
        max_epochs = max_epochs
        validation_ratio = 0.1
        x_train_split, y_train_split, x_val, y_val = self.train_validation_split(x_train, y_train, validation_ratio)
        num_samples = x_train_split.shape[0]

        momentums = {key: np.zeros_like(value) for key, value in self.weights.items()}
        momentums.update({key: np.zeros_like(value) for key, value in self.biases.items()})
        beta_momentum = 0.9

        for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"): 
            # Sampling randomly
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train_split[indices]
            y_train_shuffled = y_train_split[indices]

            total_loss = 0

            for i in range(0, num_samples, batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]/255.0
                y_batch = y_train_shuffled[i:i + batch_size]

                # Initialize accumulated gradients
                d_theta = {key: np.zeros_like(value) for key, value in self.weights.items()}
                d_theta.update({key: np.zeros_like(value) for key, value in self.biases.items()})

                batch_loss = 0

                gradients_at_parameter = {key: np.zeros_like(value) for key, value in self.gradients.items()}

                for j in range(len(x_batch)):
                    input = x_batch[j].reshape(self.input_size, 1)
                    output = y_batch[j]
                    
                    self.forward_propagation(input)
                    self.backward_propagation(input, output)
                    
                    # d_theta accumulates gradients across the batch
                    d_theta = {key: value + self.gradients[key] for key, value in d_theta.items()}                        
                    batch_loss += -np.log(self.h[f"h{self.num_layers+1}"][output, 0] + 1e-9)  # Avoid log(0)

                    if optimizer=="nag":
                        weights_temp = self.weights
                        biases_temp = self.biases
                        weights = {key: value - beta_momentum * momentums[key] for key, value in self.weights.items()}
                        biases = {key: value - beta_momentum * momentums[key] for key, value in self.biases.items()}
                        weights_temp = self.weights
                        biases_temp = self.biases
                        self.weights = weights
                        self.biases = biases
                        self.forward_propagation(input)
                        self.backward_propagation(input, output)
                        gradients_at_parameter = {key: value + self.gradients[key] for key, value in gradients_at_parameter.items()}
                        self.weights = weights_temp
                        self.biases = biases_temp


                # Average gradients over batch
                # d_theta = {key: value / len(x_batch) for key, value in d_theta.items()}

                # Update weights and biases
                if optimizer=="sgd":
                    self.weights = {key: value - self.eta * d_theta[key] for key, value in self.weights.items()}
                    self.biases = {key: value - self.eta * d_theta[key] for key, value in self.biases.items()}
                elif optimizer=="momentum":
                    momentums = {key: beta_momentum * value + d_theta[key] for key, value in momentums.items()}
                    self.weights = {key: value - self.eta * momentums[key] for key, value in self.weights.items()}
                    self.biases = {key: value - self.eta * momentums[key] for key, value in self.biases.items()}
                elif optimizer=="nag":
                    # weights = {key: value - beta_momentum * momentums[key] for key, value in self.weights.items()}
                    # biases = {key: value - beta_momentum * momentums[key] for key, value in self.biases.items()}
                    # gradients_at_parameter = self.return_gradients(weights, biases, x_batch, y_batch)
                    momentums = {key: beta_momentum * value + gradients_at_parameter[key] for key, value in momentums.items()}
                    self.weights = {key: value - self.eta * momentums[key] for key, value in self.weights.items()}
                    self.biases = {key: value - self.eta * momentums[key] for key, value in self.biases.items()}

                total_loss += batch_loss

            # grad_norms_min = min(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # grad_norms_max = max(np.linalg.norm(grad, ord=np.inf) for grad in d_theta.values())
            # print(f"Minimum of max norm of each gradient: {grad_norms_min}")
            # print(f"Maximum of max norm of each gradient: {grad_norms_max}")

            avg_loss = total_loss / num_samples  # Average loss across all batches

            accurate_predictions = 0
            for i in range(x_val.shape[0]):
                y_pred = np.argmax(self.forward_propagation(x_val[i].reshape(self.input_size,1) / 255.0))
                if y_pred==y_val[i]:
                    accurate_predictions += 1

            print(f"Epoch {epoch}/{max_epochs} - Training Loss: {avg_loss:.4f}, Validation Accuracy: {(accurate_predictions*1000)/num_samples: .4f}%")


    # def gradient_descent_general(self, x_train, y_train, batch_size=128, max_epochs=50, optimizer='adam', eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, gamma=0.9):
    #     validation_ratio = 0.1
    #     x_train_split, y_train_split, x_val, y_val = self.train_validation_split(x_train, y_train, validation_ratio)
    #     num_samples = x_train_split.shape[0]
        
    #     # Initialize optimizer-specific parameters
    #     momentums = {key: np.zeros_like(value) for key, value in self.weights.items()}
    #     momentums.update({key: np.zeros_like(value) for key, value in self.biases.items()})
    #     velocities = {key: np.zeros_like(value) for key, value in self.weights.items()}
    #     velocities_b = {key: np.zeros_like(value) for key, value in self.biases.items()}
        
    #     t = 0  # Adam and Nadam require time step

    #     for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"): 
    #         indices = np.random.permutation(num_samples)
    #         x_train_shuffled = x_train_split[indices] / 255.0
    #         y_train_shuffled = y_train_split[indices]

    #         total_loss = 0
            
    #         for i in range(0, num_samples, batch_size):
    #             x_batch = x_train_shuffled[i:i + batch_size]
    #             y_batch = y_train_shuffled[i:i + batch_size]
                
    #             d_theta = {key: np.zeros_like(value) for key, value in self.weights.items()}
    #             d_theta.update({key: np.zeros_like(value) for key, value in self.biases.items()})
                
    #             batch_loss = 0
    #             for j in range(len(x_batch)):
    #                 input = x_batch[j].reshape(self.input_size, 1)
    #                 output = y_batch[j]
                    
    #                 self.forward_propagation(input)
    #                 self.backward_propagation(input, output)
                    
    #                 for key in d_theta:
    #                     d_theta[key] += self.gradients[key]
    #                     # d_theta_b[key] += self.gradients[key.replace('W', 'b')]
                    
    #                 batch_loss += -np.log(self.h[f"h{self.num_layers+1}"][output, 0] + 1e-9)  # Avoid log(0)
                
    #             # Apply optimizer
    #             t += 1
    #             for key in self.weights:
    #                 if optimizer == 'sgd':  # Vanilla SGD
    #                     self.weights[key] -= eta * d_theta[key]
    #                     self.biases[key] -= eta * d_theta_b[key]
    #                 elif optimizer == 'momentum':  # Momentum SGD
    #                     momentums[key] = gamma * momentums[key] + eta * d_theta[key]
    #                     momentums_b[key] = gamma * momentums_b[key] + eta * d_theta_b[key]
    #                     self.weights[key] -= momentums[key]
    #                     self.biases[key] -= momentums_b[key]
    #                 elif optimizer == 'nesterov':  # NAG
    #                     prev_momentum = momentums[key]
    #                     momentums[key] = gamma * momentums[key] + eta * d_theta[key]
    #                     self.weights[key] -= -gamma * prev_momentum + (1 + gamma) * momentums[key]
    #                     self.biases[key] -= -gamma * momentums_b[key] + (1 + gamma) * momentums_b[key]
    #                 elif optimizer == 'adam':  # Adam
    #                     momentums[key] = beta1 * momentums[key] + (1 - beta1) * d_theta[key]
    #                     velocities[key] = beta2 * velocities[key] + (1 - beta2) * (d_theta[key] ** 2)
    #                     m_hat = momentums[key] / (1 - beta1 ** t)
    #                     v_hat = velocities[key] / (1 - beta2 ** t)
    #                     self.weights[key] -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
    #                     self.biases[key] -= eta * momentums_b[key] / (np.sqrt(velocities_b[key]) + epsilon)
    #                 elif optimizer == 'nadam':  # Nadam
    #                     m_hat = (beta1 * momentums[key] + (1 - beta1) * d_theta[key]) / (1 - beta1 ** t)
    #                     v_hat = velocities[key] / (1 - beta2 ** t)
    #                     self.weights[key] -= eta * (beta1 * m_hat + (1 - beta1) * d_theta[key] / (1 - beta1 ** t)) / (np.sqrt(v_hat) + epsilon)
    #                     self.biases[key] -= eta * (beta1 * momentums_b[key] + (1 - beta1) * d_theta_b[key] / (1 - beta1 ** t)) / (np.sqrt(velocities_b[key]) + epsilon)
                    
    #             total_loss += batch_loss
            
    #         avg_loss = total_loss / num_samples  # Average loss across all batches
            
    #         accurate_predictions = 0
    #         for i in range(x_val.shape[0]):
    #             y_pred = np.argmax(self.forward_propagation(x_val[i].reshape(self.input_size, 1)))
    #             if y_pred == y_val[i]:
    #                 accurate_predictions += 1

    #         print(f"Epoch {epoch}/{max_epochs} - Loss: {avg_loss:.4f}, Validation Accuracy: {(accurate_predictions * 100) / x_val.shape[0]: .4f}%")


