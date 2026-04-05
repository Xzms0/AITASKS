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

        grad_norm = grad_output * self.params['gamma']

        grad_var = grad_norm * (X - avg) * -0.5 * (var + 1e-8) ** -1.5
        grad_avg = -np.sum(grad_norm  / np.sqrt(var + 1e-8), axis=(0, 2, 3), keepdims=True) + \
            grad_var * -2 * np.mean(X - avg, axis=(0, 2, 3), keepdims=True)
        grad_input = grad_norm  / np.sqrt(var + 1e-8) + \
            grad_var * 2 * (X - avg) / X.shape[0] + grad_avg / X.shape[0]
        
        return grad_input
