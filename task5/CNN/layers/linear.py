import numpy as np

class Linear:
    def __init__(self, in_features, out_features, std=1e-3, bias=True):
        self.params = {}
        self.params['W'] = np.random.randn((in_features, out_features)) * std
        self.params['b'] = np.random.randn(out_features) * std

        self.bias = bias
        self.cache = None


    def forward(self, X):
        '''
        X: N*in_features
        y: N*out_features
        '''
        self.cache = X.copy()
        out = np.dot(X, self.params['W'])

        if self.bias:
            out += self.params['b']

        return out


    def backward(self):
        pass