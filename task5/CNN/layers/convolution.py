import numpy as np

class Conv2D:
    def __init__(self, in_chanels=3, out_chanels=16,kernel_size=3,
                 stride=1, padding=1, std=1e-3):
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params = np.random.randn(in_chanels, kernel_size, kernel_size, out_chanels) * std
        
        self.cache = None


    def _pad(self, X):
        X_copy = X.copy()
        pad1 = np.zeros((X.shape[0], X.shape[1], 1, X.shape[3]))
        X_copy = np.concatenate((pad1, X_copy, pad1), axis=2)
        pad2 = np.zeros((X.shape[0], X.shape[1], X.shape[2]+self.padding*2, 1))
        X_copy = np.concatenate((pad2, X_copy, pad2))

        return X_copy
    

    def forward(self, X):
        '''
        X: N*in_chanels*width*height
        y: N*out_chanels*width*height
        '''
        self.cache = X.copy()
        X_pad = self._pad(X)

        out_width = (X_pad.shape[2] - self.kernel_size) // self.stride + 1
        out_height = (X_pad.shape[3] - self.kernel_size) // self.stride + 1

        out = np.zeros((X.shape[0], self.out_chanels, out_width, out_height))

        for i in range(out_width):
            for j in range(out_height):
                w_start = i * self.stride
                w_end = w_start + self.kernel_size

                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                piece = X_pad[:, :, w_start:w_end, h_start:h_end]
                piece = np.tensordot(piece, self.params, axes=3)
                out[:, :, i, j] = piece

        return out


    def backward(self):
        pass