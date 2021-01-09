from src_to_implement.Layers.Base import BaseLayer
import numpy as np
from scipy import signal

class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random([num_kernels, *self.convolution_shape])
        self.bias = np.random.random(num_kernels)
        self._optimizer = None
        self._optimizer_b = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_num = input_tensor.shape[0]
        channel_num = input_tensor.shape[1]
        ONEDCONV = len(input_tensor.shape) == 3
        output_tensor = np.zeros([batch_num, self.num_kernels, *input_tensor.shape[2:]])
        
        for b in range(batch_num):            # iterate over each tensor in the batch
            for k in range(self.num_kernels): # iterate over each kernel
                for c in range(channel_num):  # iterate over each channel to sum them up in the end to get 3D convolution (feature map)
                    output_tensor[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode = 'same')
        
                output_tensor[b,k] += self.bias[k] # add bias to each feature map

        # stride 
        if ONEDCONV:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        new_error_tensor = np.zeros(self.input_tensor.shape)
        grad_weights = np.zeros(self.weights.shape)
        self.gradient_tensor = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        batch_num = error_tensor.shape[0]
        channels = self.weights.shape[1]

        for b in range(batch_num):
            
            # Stride
            error_tensor_strided = np.zeros((self.num_kernels, *self.input_tensor.shape[2:] ))
            ERR2D = len(error_tensor.shape)==4
            
            for k in range(error_tensor.shape[1]):
                errorimage = error_tensor[b, k, :]
                if ERR2D: error_tensor_strided[k,::self.stride_shape[0], ::self.stride_shape[1]] = errorimage
                else: error_tensor_strided[k,::self.stride_shape[0]] = errorimage

            # Gradient with respect to the input
            for c in range(channels):
                err = signal.convolve(error_tensor_strided, np.flip(self.weights, 0)[:,c,:], mode='same')
                midchannel = int(err.shape[0] / 2)
                op = err[midchannel,:]
                new_error_tensor[b,c,:] = op   

            # Gradient with respect to the weights
            for k in range(self.num_kernels):
                self.grad_bias[k] += np.sum(error_tensor[b, k, :]) 

                for c in range(self.input_tensor.shape[1]):
                    inputimg = self.input_tensor[b,c,:]

                    if ERR2D:
                        padx = self.convolution_shape[1]/2
                        pady = self.convolution_shape[2]/2
                        px = (int(np.floor(padx)),int(np.floor(padx-0.5)))
                        py = (int(np.floor(pady)),int(np.floor(pady-0.5)))
                        padimg = np.pad(inputimg, (px,py))
                    else:
                        padx = self.convolution_shape[1]/2
                        px = (int(np.floor(padx)),int(np.floor(padx-0.5)))
                        padimg = np.pad(inputimg, px)

                    grad_weights[k,c,:] = signal.correlate(padimg, error_tensor_strided[k,:], mode="valid")

            self.gradient_tensor += grad_weights

        # Update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        if self._optimizer_b is not None:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.grad_bias)

        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in    = np.prod(self.convolution_shape)
        fan_out   = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer_b(self):
        return self._optimizer_b

    @optimizer_b.setter
    def optimizer_b(self, optimizer_b):
        self._optimizer_b = optimizer_b

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    @property
    def gradient_bias(self):
        return self.grad_bias
