import numpy as np

class ReLU:
    def __init__(self):
        self.cache = None


    def forward(self, X):
        self.cache = X.copy()
        return np.maximum(0, X)


    def backward(self):
        pass