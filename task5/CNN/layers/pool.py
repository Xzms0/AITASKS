import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.cache = None


    def forward(self, X):
        '''
        X: (N, C, H, W)
        '''
        N, C, H, W = X.shape
        pool_size = self.pool_size
        stride = self.stride
        
        out_height = (H - pool_size) // stride + 1
        out_width = (W - pool_size) // stride + 1
        
        windows = sliding_window_view(X, (pool_size, pool_size), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        
        windows_flat = windows.reshape(N, C, out_height, out_width, -1)
        
        out = np.max(windows_flat, axis=-1)
        max_indices = np.argmax(windows_flat, axis=-1)
        
        max_pos = np.zeros((N, C, out_height, out_width, 2), dtype=np.int64)
        max_pos[:, :, :, :, 0] = max_indices // pool_size
        max_pos[:, :, :, :, 1] = max_indices % pool_size
                
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

                h_abs = h_abs.astype(int)
                w_abs = h_abs.astype(int)
                grad_input[np.arange(shape[0])[:, None], 
                        np.arange(shape[1])[None, :], 
                        h_abs, w_abs] += grad_val
    
        return grad_input


class GlobalAvgPool:
    def __init__(self):
        self.cache = None


    def forward(self, X):
        '''
        X:(N, C, H, W)
        '''
        self.cache = X.shape
        out = np.mean(X, axis=(2, 3))
        return out


    def backward(self, grad_output):
        """
        grad_output: (N, C, 1, 1)
        return: grad_input (N, C, H, W)
        """
        N, C, H, W = self.cache
        grad_input = grad_output / (H * W)
        grad_input = grad_input.reshape(N, C, 1, 1)
        grad_input = np.broadcast_to(grad_input, (N, C, H, W))
        
        return grad_input