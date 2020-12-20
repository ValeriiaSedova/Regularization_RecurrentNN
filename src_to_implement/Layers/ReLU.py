import numpy as np

class ReLU:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor[input_tensor <= 0] = 0
        self.output_tensor = input_tensor
        return self.output_tensor

    def backward(self, error_tensor):
        def relu_grad(tensor):
            tensor[tensor > 0] = 1
            tensor[tensor <= 0] = 0
            return tensor
        # gradient = np.dot(self.input_tensor.T, error_tensor)
        dydx = relu_grad(self.output_tensor)


        return dydx*error_tensor
