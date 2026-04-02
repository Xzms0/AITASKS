import numpy as np

class Solver:
    def __init__(self, model, loss_fn, optimizer, X, y, 
                 epochs=1000, batch_size=1000, 
                 verbose=True, print_every=1):
        """
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        X, y: 训练数据
        epochs: 训练轮数
        batch_size: 批大小
        verbose: 是否打印信息
        print_every: 每隔多少轮打印一次
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
        
        # 记录历史
        self.losses = []
    
    def train(self):
        n_samples = self.X.shape[0]
        n_batches = n_samples // self.batch_size if n_samples >= self.batch_size else 1
        
        print(f"Training: {self.epochs} epochs, {n_batches} batches, lr={self.optimizer.lr}\n")
        
        for epoch in range(self.epochs):
            # 打乱数据
            idx = np.random.permutation(n_samples)
            X_shuffled = self.X[idx]
            y_shuffled = self.y[idx]
            
            epoch_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # 前向传播
                out = self.model.forward(X_batch)
                loss = self.loss_fn.forward(out, y_batch)
                epoch_loss += loss
                
                # 反向传播
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                
                # 更新参数（使用优化器）
                self.optimizer.update(self.model)
            
            avg_loss = epoch_loss / n_batches
            self.losses.append(avg_loss)
            
            # 打印信息
            if self.verbose and (epoch + 1) % self.print_every == 0:
                print(f"Epoch [{epoch+1:3d}/{self.epochs}] - Loss: {avg_loss:.6f}")
        
        print(f"\nFinal Loss: {self.losses[-1]:.6f}")
        
        return self.losses
    
    def plot_loss(self):
        """绘制损失曲线"""
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.show()