from layers import activation, convolution, dropout, linear, loss, pool, normalization
from utils import manager

class CNN:
    def __init__(self):
        
        self.model = [
            convolution.Conv2D(in_chanels=3, out_chanels=8, kernel_size=3, stride=1, padding=1),
            normalization.BatchNorm(num_features=8),
            activation.ReLU(),
            pool.MaxPool(pool_size=2, stride=2),  # 8x16x16
            
            convolution.Conv2D(in_chanels=8, out_chanels=16, kernel_size=3, stride=1, padding=1),
            normalization.BatchNorm(num_features=16),
            activation.ReLU(),
            pool.MaxPool(pool_size=2, stride=2),  # 16x8x8
            
            #convolution.Conv2D(in_chanels=32, out_chanels=64, kernel_size=3, stride=1, padding=1),
            #normalization.BatchNorm(num_features=64),
            #activation.ReLU(),
            #pool.MaxPool(pool_size=2, stride=2),  # 64x4x4

            #pool.GlobalAvgPool(),
            linear.Flatten(),
            linear.Linear(in_features=16*8*8, out_features=128),
            activation.ReLU(),
            dropout.Dropout(0.5),
            linear.Linear(in_features=128, out_features=10)
        ]
        
        self.loss_fn = loss.SoftmaxCrossEntropy()
    

    def forward(self, X):
        #print(f"Original Shape: {X.shape}")
        out = X
        for layer in self.model:
            #manager.data_monitor(out)
            out = layer.forward(out)
            #print(f"{layer.__class__.__name__} Shape: {out.shape}")
        return out
    

    def loss(self, X, y):
        scores = self.forward(X)
        loss = self.loss_fn.forward(scores, y)
        return scores, loss
    

    def backward(self, grad=None):
        if grad is None:
            grad = self.loss_fn.backward()
        
        for layer in reversed(self.model):
            grad = layer.backward(grad)
        
        return grad
    

    def parameters(self):
        params_list = []
        for layer in self.model:
            if hasattr(layer, 'parameters'):
                params_list.extend(layer.parameters())
        return params_list