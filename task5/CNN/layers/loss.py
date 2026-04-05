import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None


    def forward(self, X, y):
        '''
        X: (N,C)
        y: (N,)
        '''
        N, C = X.shape
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X - X_max)
        X_prob = X_exp / np.sum(X_exp, axis=1, keepdims=True)

        self.cache = (X_prob, y)

        loss = np.mean(-np.log(X_prob[range(N), y] + 1e-8))
        return loss


    def backward(self):
        X_prob, y = self.cache
        N, C = X_prob.shape
        grad = X_prob.copy()
        grad[np.arange(N), y] -= 1

        grad_input = grad / N
        return grad_input