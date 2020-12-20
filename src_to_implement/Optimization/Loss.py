import numpy as np
from time import sleep

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        loc = input_tensor[label_tensor == 1]

        self.loss = - np.log(loc + np.finfo(float).eps)
        self.loss = self.loss.sum()
        return self.loss

    def backward(self, label_tensor):
        error_tensor = - label_tensor / self.input_tensor
        return error_tensor 