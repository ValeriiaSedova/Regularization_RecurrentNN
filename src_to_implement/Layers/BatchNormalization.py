from .Base import BaseLayer
import numpy as np
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, *args):
        super().__init__()
        self.batch_size=None
        self.num_pixels = None
        self.batch_means = 0
        self.means = 0
        self.batch_variances = 0
        self.variances = 0
        self.test_values_set = False
        self.weights = None
        self.weights_optimizer = None
        self.bias_optimizer = None
        self.bias = None
        self.epsilon = 1e-11
        self.input_tensor = None
        self.input_tensor_normed = None
        self.moving_avg_decay = 0.8
        self.channels = args[0] if len(args) is not 0 else 0
        # ----------------------------------------------------
        # Gradients
        self.gradient_weights = None
        self.gradient_gamma = None
        self.gradient_beta = None
        self.gradient_input = None
        self.delta = 1.0

    def forward(self, input_tensor):
        if self.test_values_set is not False:
            self.means = 0
            self.variances = 0
            self.test_values_set = False

        self.input_tensor = input_tensor  # Saving globally
        self.batch_size = input_tensor.shape[0]

        if self.channels != 0:  # Convolutional
            self.num_pixels = int(input_tensor.shape[1]/self.channels)
            self.input_tensor = self.input_tensor.reshape(self.batch_size * self.num_pixels, self.channels)

        if self.weights is None and self.bias is None:
            self.weights = np.ones((self.input_tensor.shape[1]))
            self.bias = np.zeros((self.input_tensor.shape[1]))

        if self.testing_phase == True:  # For testing
            # Calculate first part of moving average using old (k-1) values
            self.means += (1.0 - self.moving_avg_decay) * self.batch_means
            self.variances += (1.0 - self.moving_avg_decay) * self.batch_variances
            self.test_values_set = True
        if self.testing_phase == False:
            # Update mini-batch values
            self.batch_means = np.mean(self.input_tensor, axis=0)
            self.batch_variances = np.var(self.input_tensor, axis=0)
        if self.testing_phase == True:  # For testing
            # Calculate second part of moving average using updated (k) values
            self.means += self.moving_avg_decay * self.batch_means
            self.variances += self.moving_avg_decay * self.batch_variances

        # For Training- and Validation-Time
        if self.testing_phase == True:  # For Test-Time
            self.input_tensor_normed = (self.input_tensor - self.means) / (np.sqrt(self.variances + self.epsilon))
        else:
            self.input_tensor_normed = (self.input_tensor - self.batch_means) / \
                                       (np.sqrt(self.batch_variances + self.epsilon))

        save = self.weights * self.input_tensor_normed + self.bias

        if self.channels != 0:  # Convolutional
            self.input_tensor_normed = self.input_tensor_normed.reshape(self.batch_size * self.num_pixels, self.channels)
            save = save.reshape(self.batch_size, self.channels*self.num_pixels)

        return save

    def backward(self, error_tensor):
        if self.channels != 0:  # Convolutional
            error_tensor = error_tensor.reshape(self.batch_size*self.num_pixels, self.channels)
        batch_size = error_tensor.shape[0]

        # Gradient w.r.t. weights
        self.gradient_gamma = np.sum(self.input_tensor_normed * error_tensor, axis=0)  # TODO: Ordering?

        # Gradient w.r.t. bias
        self.gradient_beta = np.sum(error_tensor, axis=0)

        # Gradient w.r.t. input TODO: change "Helpers"
        gradient_input_normed = error_tensor * self.weights
        gradient_variances = np.sum(gradient_input_normed * (self.input_tensor - self.batch_means) * (-.5) * (self.batch_variances + self.epsilon)**(-1.5), axis=0)
        gradient_means = np.sum(gradient_input_normed * -1 / (np.sqrt(self.batch_variances + self.epsilon)), axis=0)
        self.gradient_input = gradient_input_normed * (1 / np.sqrt(self.batch_variances + self.epsilon)) + \
            gradient_variances * (2*(self.input_tensor - self.batch_means) / batch_size) + \
            gradient_means / batch_size

        # Update
        if self.weights_optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.delta, self.weights, self.gradient_gamma)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self.gradient_beta)

        # Convolutional
        if self.channels != 0:
            self.gradient_input = self.gradient_input.reshape((self.batch_size, self.channels*self.num_pixels))

        return self.gradient_input

    def get_bias(self):
        return self.bias

    def get_weights(self):
        return self.weights

    # Returns the gradient w.r.t. the weights after being calculated in the backward-pass
    def get_gradient_weights(self):
        return self.gradient_gamma

    # Returns the gradient w.r.t. the bias after being calculated in the backward-pass
    def get_gradient_bias(self):
        return self.gradient_beta

    # Stores the optimizer for this layer
    # Note: Two optimizers might be necessary if bias and weights are handled separately
    def set_optimizer(self, optimizer):
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    # Reinitializes the weights and bias
    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
