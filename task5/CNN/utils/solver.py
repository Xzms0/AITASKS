import numpy as np
from utils import manager

class Solver:
    def __init__(self, model, loss_fn, optimizer, X, y, 
                 epochs=1000, batch_size=1000, 
                 verbose=True, print_every=1, decay_every=None):
        """
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        X, y: 训练数据
        epochs: 训练轮数
        batch_size: 批大小
        verbose: 是否打印信息
        print_every: 每隔多少轮打印一次
        decay_every: 学习率多少轮衰减一次
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.X = X
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.print_every = print_every
        self.decay_every = decay_every
        
        # 记录历史
        self.losses = []
    

    def train(self):
        manager.set_mode('train')
        n_samples = self.X.shape[0]
        n_batches = n_samples // self.batch_size
        
        print(f"Training: {self.epochs} epochs, {n_batches} batches, lr={self.optimizer.lr}")
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(0, n_samples, self.batch_size):

                print(f'{epoch+1}/{self.epochs} epochs,',\
                      f'{i//self.batch_size+1}/{n_batches} batches', end='\r')

                X_batch = self.X[i:i+self.batch_size]
                y_batch = self.y[i:i+self.batch_size]
                
                # 前向传播
                out = self.model.forward(X_batch)
                acc  = np.mean(np.argmax(out, axis=1) == y_batch)
                loss = self.loss_fn.forward(out, y_batch)
                epoch_loss += loss
                epoch_acc += acc
                
                # 反向传播
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                
                # 更新参数（使用优化器）
                self.optimizer.update(self.model)

            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
            self.losses.append(avg_loss)
            
            if self.decay_every and (epoch + 1) % self.decay_every == 0:
                self.optimizer.decay()
                
            # 打印信息
            if self.verbose and (epoch + 1) % self.print_every == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}, Acc: {avg_acc*100:.2f}%")
        
        print(f"Final Loss: {self.losses[-1]:.6f}")
        
        return self.losses
    

    def evaluate(self, X, y):
        manager.set_mode('eval')
        correct = 0
        n = X.shape[0]
        
        for i in range(0, n, self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]
            
            scores = self.model.forward(X_batch)
            pred = np.argmax(scores, axis=1)
            correct += np.sum(pred == y_batch)
    
        return correct / n


    def plot_loss(self):
        """绘制损失曲线"""
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.show()