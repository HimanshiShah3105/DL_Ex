from Base import *
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True
        weights = np.random.rand(self.input_size, self.output_size) # not sure if this is right 
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)
        
    def set_opt(self):
        pass
    
    def get_opt(self):
        pass
    
    optimizer = property(get_opt, set_opt)
    
    def backward(self, error_tensor):
        pass
        
        