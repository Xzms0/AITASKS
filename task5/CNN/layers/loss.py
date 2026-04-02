import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None


    def parameters(self):
        pass


    def forward(self, X, y):
        '''
        X: N*C
        y: N*C
        '''
        
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X - X_max)
        X_prob = X_exp / np.sum(X_exp, axis=1, keepdims=True)

        self.cache = (X_prob, y)

        loss = np.mean(-np.log(X_prob[range(X.shape[0]), y] + 1e-8))
        return loss


    def backward(self):
        X_prob, y = self.cache
        grad_input =  X_prob - y
        return grad_input