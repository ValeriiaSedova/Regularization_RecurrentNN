import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random([self.input_size + 1, self.output_size])
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        height, width = input_tensor.shape
        self.input_tensor = np.concatenate((input_tensor, np.ones([height, 1])), axis = 1)
        y_hat = np.dot(self.input_tensor, self.weights)
        return y_hat 

    def backward(self, error_tensor):
        error_tensor_new = np.dot( error_tensor, self.weights[:-1,:].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer != None: 
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)        
        return error_tensor_new

    def initialize(self, weights_initializer, bias_initializer):
        fan_in  = np.prod(self.input_size)
        fan_out = np.prod(self.output_size)
        self.weights[:-1,:] = weights_initializer.initialize(self.weights[:-1,:].shape, fan_in, fan_out)
        self.weights[-1, :] = bias_initializer.initialize(self.weights[-1].shape,fan_in, fan_out)
        # self.weights = np.concatenate((self.weights, bias), axis = 0)


    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights


    