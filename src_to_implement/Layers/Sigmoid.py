import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1.0 / (1.0 + np.exp(-1.0 * input_tensor))
        return self.activation

    def backward(self, error_tensor):
        derivative = self.activation * (1 - self.activation)
        return derivative * error_tensor
