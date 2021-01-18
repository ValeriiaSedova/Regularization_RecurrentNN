import numpy as np

class Optimizer:
        
    def __init__(self):
        self.regularizer = None
        self.learning_rate = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def reg(self, weights):
        if self.regularizer != None:
            return weights - self.learning_rate * self.regularizer.calculate_gradient(weights)
        else:
            return weights


class Sgd(Optimizer):

    def __init__(self, learning_rate:float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return self.reg(weight_tensor) - self.learning_rate * gradient_tensor

class SgdWithMomentum(Optimizer):
    
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.momentum_rate * self.vk - self.learning_rate * gradient_tensor
        return self.reg(weight_tensor) + self.vk

class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0
        self.rk = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.mu * self.vk + (1 - self.mu) * gradient_tensor
        self.rk = self.rho * self.rk + (1 - self.rho) * gradient_tensor * gradient_tensor
        vk_hat = self.vk / (1 - self.mu**self.k)  
        rk_hat = self.rk / (1 - self.rho**self.k)
        self.k += 1
        return self.reg(weight_tensor) - self.learning_rate * (vk_hat / (np.sqrt(rk_hat) + np.finfo(np.float64).eps))



