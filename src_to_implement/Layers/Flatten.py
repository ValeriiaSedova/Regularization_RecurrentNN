from .Base import BaseLayer

class Flatten(BaseLayer):
    
    def __init__(self):
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        b, c, x, y = self.input_shape
        output_tensor = input_tensor.reshape(b, c*x*y)
        return output_tensor
        
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)