import numpy as np
from utils import manager

class BatchNorm:
    def __init__(self, num_features=3, momentum=0.9):
        self.momentum = momentum

        self.params = {}
        self.params['gamma'] = np.ones((1, num_features, 1, 1))
        self.params['beta'] = np.zeros((1, num_features, 1, 1))

        self.grad = {}

        self.running_avg = np.ones((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.cache = None


    def parameters(self):
        return [(self.params['gamma'], self.grad['gamma']), (self.params['beta'], self.grad['beta'])]


    def forward(self, X):
        '''
        X: (N, C, H, W)
        '''
        if manager.RunningMode == 'train':
            avg = np.mean(X, axis=(0, 2, 3), keepdims=True)
            var = np.var(X, axis=(0, 2, 3), keepdims=True)

            self.running_avg = self.momentum * self.running_avg + \
                                (1 - self.momentum) * avg
            self.running_var = self.momentum * self.running_var + \
                               (1 - self.momentum) * var

        else:
            avg = self.running_avg
            var = self.running_var

        X_norm = (X - avg) / np.sqrt(var + 1e-8)
        
        self.cache = (X, X_norm, avg, var)
        out = X_norm * self.params['gamma'] + self.params['beta']
        return out


    def backward(self, grad_output):
        '''
        grad_output: (N, C, H, W)
        '''
        X, X_norm, avg, var = self.cache

        self.grad['gamma'] = np.sum(grad_output * X_norm, axis=(0, 2, 3), keepdims=True)
        self.grad['beta'] = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)

        N, C, H, W = X.shape
        gamma = self.params['gamma']

        # 计算中间变量
        std = np.sqrt(var + 1e-8)
        X_centered = X - avg
        dX_norm = grad_output * gamma

        term1 = dX_norm / std
        term2 = -np.mean(dX_norm, axis=(0, 2, 3), keepdims=True) / std
        term3 = -X_norm * np.mean(dX_norm * X_norm, axis=(0, 2, 3), keepdims=True) / std

        grad_input = term1 + term2 + term3

        return grad_input
        
