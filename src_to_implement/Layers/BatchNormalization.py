import numpy as np
from Layers import Base
from Layers import Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mu = 0
        self.sigma = 0
        self.initialize('place','holder')
        self._optimizer = None
        self._optimizerbias = None

    def initialize(self, place, holder):
        place = None; holder = None
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            batch, channel, height, width = tensor.shape
            
            output_tensor = np.reshape(tensor, [batch, channel, height * width])
            output_tensor = np.transpose(output_tensor, [0, 2, 1])
            output_tensor = np.reshape(output_tensor, [batch * height * width, channel])
        
        else:
            batch, channel, height, width  = self.input_tensor.shape

            output_tensor = np.reshape(tensor, [batch, height * width, channel])
            output_tensor = np.transpose(output_tensor, [0, 2, 1])
            output_tensor = np.reshape(output_tensor, [batch,channel, height, width])          

        return output_tensor
    
    def forward(self, input_tensor):
        conv_shape = len(input_tensor.shape) == 4 # Define the condition based on shape
        self.input_tensor = input_tensor
        
        if conv_shape: self.reformatted_tensor = self.reformat(self.input_tensor)
        else:          self.reformatted_tensor = self.input_tensor

        if not self.testing_phase:
            self.mu_B = np.mean(self.reformatted_tensor, axis = 0)
            self.sigma_B = np.std(self.reformatted_tensor, axis = 0)
            self.x_hat = (self.reformatted_tensor - self.mu_B) / np.sqrt(self.sigma_B**2 + np.finfo(float).eps)
            self.y_hat = self.weights * self.x_hat + self.bias

            # Moving average decay:
            alpha = 0.8
            self.mu = alpha * self.mu + (1 - alpha) * self.mu_B
            self.sigma = alpha * self.sigma + (1 - alpha) * self.sigma_B
        
        else:
            self.x_hat = (self.reformatted_tensor - self.mu) / np.sqrt(self.sigma**2 + np.finfo(float).eps)
            self.y_hat = self.weights * self.x_hat + self.bias

        if conv_shape: self.y_hat = self.reformat(self.y_hat)

        return self.y_hat

    def backward(self, error_tensor):
        conv_shape = len(error_tensor.shape) == 4

        if conv_shape: self.error_tensor = self.reformat(error_tensor)
        else: self.error_tensor = np.reshape(error_tensor,self.x_hat.shape)

        gradient_weights = np.sum(self.error_tensor * self.x_hat, axis = 0)
        self.gradient_weights = np.reshape(gradient_weights, [1, self.channels])

        gradient_bias = np.sum(self.error_tensor, axis = 0)
        self.gradient_bias = np.reshape(gradient_bias, [1, self.channels])

        if self._optimizer is not None:     self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._optimizerbias is not None: self.bias = self._optimizerbias.calculate_update(self.bias, self.gradient_bias)

        self.gradient_input = Helpers.compute_bn_gradients(
            self.error_tensor,
            self.reformatted_tensor,
            self.weights,
            self.mu_B,
            self.sigma_B**2,
            np.finfo(float).eps)

        if conv_shape: self.gradient_input = self.reformat(self.gradient_input)

        return self.gradient_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self._optimizerbias = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias