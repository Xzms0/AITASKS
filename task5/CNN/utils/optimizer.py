import numpy as np

class SGD:
    def __init__(self, lr=0.01, decay_rate=0.95):
        self.lr = lr
        self.decay_rate = decay_rate
    
    def update(self, model):
        for param, grad in model.parameters():
            param -= self.lr * grad

    
    def decay(self):
        self.lr *= self.decay_rate



#By AI
class SGDWithMomentum:
    """带动量的SGD"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, model):
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(param)
            
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            param += self.velocities[i]


class Adam:
    """Adam优化器"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, model):
        self.t += 1
        
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)