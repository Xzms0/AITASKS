import numpy as np

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.cache = None


    def parameters(self):
        pass


    def forward(self, X):
        '''
        X: (N, C, H, W)
        '''
        out_height = (X.shape[2] - self.pool_size) // self.stride + 1
        out_width = (X.shape[3] - self.pool_size) // self.stride + 1
        out = np.zeros((X.shape[0], X.shape[1], out_height, out_width))
        max_pos = np.zeros((X.shape[0], X.shape[1], out_height, out_width, 2))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                    
                window = X[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

                window_flat = window.reshape(X.shape[0], X.shape[1], -1)
                max_index = np.argmax(window_flat, axis=2)
                max_pos[:, :, i, j, 0] = max_index // self.pool_size
                max_pos[:, :, i, j, 1] = max_index % self.pool_size
                
        self.cache = (X.shape, max_pos)
        return out


    def backward(self, grad_output):
        '''
        grad_output: (N, C, H_out, W_out)
        grad_input: (N, C, H_in, W_in)
        '''
        shape, max_pos = self.cache
        _, _, h_out, w_out = grad_output.shape
        
        grad_input = np.zeros(shape)
        
        h_starts = np.arange(h_out) * self.stride
        w_starts = np.arange(w_out) * self.stride
        
        max_h = max_pos[:, :, :, :, 0]
        max_w = max_pos[:, :, :, :, 1]
        
        for i in range(h_out):
            for j in range(w_out):
                h_abs = h_starts[i] + max_h[:, :, i, j]
                w_abs = w_starts[j] + max_w[:, :, i, j]
                grad_val = grad_output[:, :, i, j]

                grad_input[np.arange(shape[0])[:, None], 
                        np.arange(shape[1])[None, :], 
                        h_abs, w_abs] += grad_val
    
        return grad_input


class GlobalAvgPool:
    def __init__(self):
        self.cache = None


    def parameters(self):
        pass


    def forward(self, X):
        '''
        X:(N, C, H, W)
        '''
        self.cache = X.shape
        out = np.mean(X, axis=(2, 3), keepdims=True)
        return out


    def backward(self, grad_output):
        """
        grad_output: (N, C, 1, 1)
        return: grad_input (N, C, H, W)
        """
        N, C, H, W = self.cache
        grad_input = grad_output / (H * W)
        grad_input = np.broadcast_to(grad_input, (N, C, H, W))
        
        return grad_input