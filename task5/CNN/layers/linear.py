import numpy as np

class Linear:
    def __init__(self, in_features, out_features, std=1e-3, bias=True):
        self.params = {}
        self.params['W'] = np.random.randn(in_features, out_features) * std
        self.params['b'] = np.random.randn(out_features) * std
        
        self.grad = {}
        self.grad['W'] = None
        self.grad['b'] = None
        
        self.bias = bias
        self.cache = None
    

    def parameters(self):
        params_list = [(self.params['W'], self.grad['W'])]
        if self.bias:
            params_list.append((self.params['b'], self.grad['b']))
        return params_list
    

    def forward(self, X):
        """
        X: (N, in_features)

        return: (N, out_features)
        """
        self.cache = X
        out = X @ self.params['W']
        if self.bias:
            out += self.params['b']
        return out
    
    def backward(self, grad_output):
        """
        grad_output: (N, out_features)

        return: grad_input (N, in_features)
        """
        self.grad['W'] = self.cache.T @ grad_output
        
        if self.bias:
            self.grad['b'] = grad_output.sum(axis=0)
        
        grad_input = grad_output @ self.params['W'].T
        return grad_input