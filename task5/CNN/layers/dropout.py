import numpy as np
from utils import manager

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.cache = None
    

    def forward(self, X):
        if manager.RunningMode == 'train':
            prob = 1 - self.p
            mask = np.random.binomial(1, prob, size=X.shape)
            out = X * mask / prob

        else:
            out = X
        
        self.cache = X, mask
        return out
    
    
    def backward(self, grad_output):
        X, mask = self.cache
        prob = 1 - self.p
        grad_input = grad_output * mask / prob
        
        return grad_input