import numpy as np

class ReLU:
    def __init__(self):
        self.cache = None


    def parameters(self):
        pass


    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)


    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.cache <= 0] = 0
        return grad_input