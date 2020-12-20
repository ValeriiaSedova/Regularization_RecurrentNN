import copy

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        activation = self.input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)
        return self.loss_layer.forward(activation, self.label_tensor)

    def backward(self):
        back_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            back_tensor = layer.backward(back_tensor)
        return back_tensor
        
    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer_b = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            error_tensor = self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        prediction = input_tensor
        for layer in self.layers:
            prediction = layer.forward(prediction)
        return prediction

    