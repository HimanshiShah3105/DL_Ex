from Layers.Base import *
from Optimization.Optimizers import Sgd
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(size = (self.input_size+1, self.output_size), low = 0, high = 1)
    def forward(self, input_tensor):
        self.input_tensor = np.c_[ input_tensor, np.ones(input_tensor.shape[0]) ]    
        self.output=np.dot(self.input_tensor,self.weights)
        return self.output
        
    def set_opt(self):
        pass
    
    def get_opt(self):
        pass
    
    optimizer = property(get_opt, set_opt)
    
    def backward(self, error_tensor):
        print(error_tensor.shape)
        input_error = np.dot(error_tensor, self.weights.T)
        gradient_tensor = np.dot(self.input_tensor.T, error_tensor)
        # self.weights[:-1] -= self.learning_rate * weights_error
        # # print(self.weights[self.input_size:,:].shape)
        # self.weights[self.input_size:,:]-= self.learning_rate * error_tensor[-1:,:]
        self.weights=Sgd.calculate_update(self.weights,gradient_tensor)
        return input_error
    # gradient_weights=