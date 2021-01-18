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
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))
        self._optimizer = None
        self._optimizerbias = None


    def initialize(self):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))


    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            batch = tensor.shape[0]
            channel = tensor.shape[1]
            height = tensor.shape[2]
            width = tensor.shape[3]

            tensor2 = np.reshape(tensor, (batch, channel, height * width))
            tensor2 = np.transpose(tensor2, (0, 2, 1))
            tensor2 = np.reshape(tensor2, (batch * height * width, channel))
        else:
            batch = self.input_tensor.shape[0]
            channel = self.input_tensor.shape[1]
            height = self.input_tensor.shape[2]
            width = self.input_tensor.shape[3]

            tensor2 = np.reshape(tensor, (batch, height * width, channel))
            tensor2 = np.transpose(tensor2, (0, 2, 1))
            tensor2 = np.reshape(tensor2, (batch,channel, height, width))          

        return tensor2

    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 4:
            self.input_tensor2 = self.reformat(self.input_tensor)
        else:
            self.input_tensor2 = self.input_tensor

        #self.input_tensor2 = self.reformat(self.input_tensor)


        if self.testing_phase == False:
            self.mu_B = np.mean(self.input_tensor2, axis=0)
            self.sigma_B = np.std(self.input_tensor2, axis=0)
            self.x_hat = (self.input_tensor2 - self.mu_B)/(self.sigma_B**2 + np.finfo(float).eps) **0.5
            self.y_hat = self.weights*self.x_hat + self.bias

            #for testing
            alpha = 0.8
            self.mu = alpha* self.mu + (1 - alpha)* self.mu_B
            self.sigma = alpha* self.sigma + (1 - alpha)* self.sigma_B
        
        else:
            self.x_hat = (self.input_tensor2 - self.mu)/(self.sigma**2 + np.finfo(float).eps) **0.5
            self.y_hat = self.weights*self.x_hat + self.bias

        if len(self.input_tensor.shape) == 4:
            self.y_hat = self.reformat(self.y_hat)
        return self.y_hat




    def backward(self, error_tensor):
        #self.error_tensor = error_tensor
        if len(error_tensor.shape) == 4:
            self.error_tensor = self.reformat(error_tensor)

        else:
            self.error_tensor = np.reshape(error_tensor,self.x_hat.shape)

        #gradient w.r.t. weights
        gradient_weights = np.sum(self.error_tensor * self.x_hat, axis=0)
        self.gradient_weights = np.reshape(gradient_weights, (1,self.channels))

        #gradient w.r.t. biases
        gradient_bias = np.sum(self.error_tensor, axis=0)
        self.gradient_bias = np.reshape(gradient_bias, (1,self.channels))

        #gradient w.r.t. inputs
        self.gradient_input = Helpers.compute_bn_gradients(self.error_tensor, self.input_tensor2, self.weights, self.mu_B, self.sigma_B**2, np.finfo(float).eps)

        #update
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._optimizerbias is not None:
            self.bias = self._optimizerbias.calculate_update(self.bias, self.gradient_bias)

        if len(error_tensor.shape) == 4:
           self.gradient_input = self.reformat(self.gradient_input)

        return self.gradient_input


    # optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)
        self._optimizerbias = copy.deepcopy(value)

    # gradient_weights property
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    # gradient_bias property
    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value