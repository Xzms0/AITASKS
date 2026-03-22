import numpy as np

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.cache = None


    def forward(self, X):
        '''
        X: N*channels*width*height
        '''
        out_width = (X.shape[2] - self.pool_size) // self.stride - 1
        out_height = (X.shape[3] - self.pool_size) // self.stride - 1
        out = np.zeros((X.shape[0], X.shape[1], out_width, out_height))
        max_pos = np.zeros((X.shape[0], X.shape[1], out_width, out_height, 2))

        for i in range(out_width):
            for j in range(out_height):
                w_start = i * self.stride
                w_end = w_start + self.pool_size
                h_start = j * self.stride
                h_end = h_start + self.pool_size
                    
                window = X[:, :, w_start:w_end, h_start:h_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

                window_flat = window.reshape(X.shape[0], X.shape[1], -1)
                max_index = np.argmax(window_flat, axis=2)
                max_w = max_index // self.pool_size
                max_h = max_index % self.pool_size
                max_pos[:, :, i, j, 0] = max_w
                max_pos[:, :, i, j, 1] = max_h
                
        self.cache = (X, max_pos, self.pool_size, self.stride)
        return out


    def backward(self):
        pass


class AvgPool:
    def __init__(self):
        self.cache = None


    def forward(self, X):
        '''
        X:N*channels*width*height
        '''
        self.cache = X.shape
        out = np.mean(X, axis=(2, 3), keepdims=True)
        return out


    def backward(self):
        pass