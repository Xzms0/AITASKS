from layers import activation, convolution, dropout, linear, loss, pool
from utils import loader, optimizer, solver

class CNN:
    def __init__(self):
        self.model = [convolution.Conv2D(), activation.ReLU(), 
                  pool.AvgPool(), linear.Linear(in_features=16, out_features=32), 
                  activation.ReLU(), linear.Linear(in_features=32, out_features=10)]
        
        self.loss = loss.Softmax()
        
    
    def forward(self, X, y):
        in_data = X.copy()
        for layer in self.model:
            in_data = layer.forward(in_data)

        scores = in_data.copy()
        loss = self.loss.forward(in_data, y)
        return scores, loss
    
    
    def backward(self):
        pass