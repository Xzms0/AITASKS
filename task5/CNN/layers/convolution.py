import numpy as np

from utils import im2col

class Conv2D:
    def __init__(self, in_chanels=3, out_chanels=16,kernel_size=3,
                 stride=1, padding=1, std=1e-3):
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.params['W'] = np.random.randn(out_chanels, in_chanels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_chanels * kernel_size * kernel_size))

        self.grad = {}
        
        self.cache = None


    def  parameters(self):
        return [(self.params['W'], self.grad['W'])]
    

    def forward(self, X):
        '''
        X: (N, in_chanels, height, width)
        return: (N, out_chanels, height, width)
        '''
        pad = ((0, 0),(0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding))
        
        X_pad = np.pad(X, pad, mode='constant')
        self.cache = X_pad

        X_col = im2col.to_col(X_pad, self.kernel_size, self.stride)
        N, C, D, L = X_col.shape
        X_col = X_col.reshape(N, -1, L) #(N, C_in*D, L)

        W_col = self.params['W'].reshape(self.out_chanels, -1) #(C_out, C_in*D)
        y = W_col @ X_col #(N, out_chanels, L)

        
        out_height = (X_pad.shape[2] - self.kernel_size) // self.stride + 1
        out_width = (X_pad.shape[3] - self.kernel_size) // self.stride + 1
        y = y.reshape(N, self.out_chanels, out_height, out_width)
        return y


    def backward(self, grad_output):
        '''
        grad_output: (N, C_out, H_out, W_out)
        '''
        X_pad = self.cache
        N, C_out, H_out, W_out = grad_output.shape
        X_col = im2col.to_col(X_pad, self.kernel_size, self.stride) #(N, C_in, L, D)
        X_col = X_col.transpose(0, 2, 1, 3).reshape(N, H_out*W_out, -1) #(N, L, C_in*D)

        G_col = grad_output.reshape(N, C_out, -1) #(N, C_out, L)
        grad_W = (G_col @ X_col).reshape(N, C_out, self.in_chanels,\
                                         self.kernel_size, self.kernel_size)
        grad_W = np.sum(grad_W, axis=0)

        self.grad['W'] = grad_W #(C_out, C_in, K, K)

        N, C_in, H_in, W_in = X_pad.shape
        W_col = self.params['W'].reshape(C_out, -1) #(C_out, C_in*D)
        grad_X = W_col.T @ G_col #(N, C_in*D, H_out*W_out)

        grad_input = im2col.to_im(grad_X, (N, C_in, H_in, W_in),\
                                  self.kernel_size, self.stride) #(N, C_in, H, W)
        
        grad_input = grad_input[:, :, self.padding:H_in-self.padding,
                                self.padding:W_in-self.padding]
        return grad_input