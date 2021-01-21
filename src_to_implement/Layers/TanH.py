import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation
    
    # Set the activation from RNN list (save them)
    def backward(self, error_tensor):
        derivative = 1 - self.activation**2
        return derivative * error_tensor



