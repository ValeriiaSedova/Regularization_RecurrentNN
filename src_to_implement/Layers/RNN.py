import numpy as np
from copy import deepcopy
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Base import BaseLayer
from Layers.Sigmoid import Sigmoid

class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False
        self.hidden_state_prev = None
        self.optimizer = None
        self._weights = None
        self.layers = [
            FullyConnected(input_size + hidden_size + 1, hidden_size),
            TanH(),
            FullyConnected(hidden_size, output_size),
            Sigmoid() ]

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # ----- Output Tensor and Hidden state ----- #
        output_tensor = np.zeros([batch_size, self.output_size])
        for time_step in range(batch_size):

            vec = input_tensor[time_step]
            tensor = np.concatenate([vec, self.hidden_state, [1]])
            tensor = np.expand_dims(tensor, axis=0)

            loc = self.layers[0].forward(tensor)
            self.hidden_state = self.layers[1].forward(loc)

            loc = self.layers[2].forward(self.hidden_state)
            output = self.layers[3].forward(loc)
            self.hidden_state = self.hidden_state.flatten()

            output_tensor[time_step] = output

        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        output_tensor = np.zeros([batch_size, self.input_size+self.hidden_size+1])
        grad_weights2 = np.zeros_like(self.layers[2].weights)
        grad_weights0 = np.zeros_like(self.layers[0].weights)
        for time_step in reversed(range(batch_size)):
            error = error_tensor[time_step]
            error = self.layers[3].backward(error)
            error = self.layers[2].backward(error)
            error = self.layers[1].backward(error)
            error = self.layers[0].backward(error)

            grad_weights2 += self.layers[2].gradient_weights
            grad_weights0 += self.layers[0].gradient_weights
            output_tensor[time_step] = error
            
        self.gradient_weights = self.layers[2].gradient_weights
        if self.optimizer is not None: 
            self.layers[2].optimizer.calculate_update(self.layers[2].weights, grad_weights2)
            self.layers[0].optimizer.calculate_update(self.layers[0].weights, grad_weights0)

        return output_tensor[:,:self.input_size]
    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.layers[0].initialize(weights_initializer, bias_initializer)
        self.layers[2].initialize(weights_initializer, bias_initializer)


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)
        self._optimizerbias = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        self._gradient_weights = self.layers[2].gradient_weights
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        # self.layers[0].gradient_weights = gradient_weights
        self._gradient_weights = gradient_weights
    
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
    # @property
    # def gradient_bias(self):
    #     return self._gradient_bias
    
    # @gradient_bias.setter
    # def gradient_bias(self, gradient_bias):
    #     self._gradient_bias = gradient_bias