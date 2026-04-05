import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_col(X, size, stride):
    '''
    X: (N, in_channels, height, width)
    return: (N, in_channels, size*size, H_out*W_out)
    '''
    N, C, H, W = X.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    
    windows = sliding_window_view(X, window_shape=(size, size), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    
    cols = windows.reshape(N, C, H_out*W_out, size*size)
    cols = cols.transpose(0, 1, 3, 2)
    
    return cols



def to_im(X_col, input_shape, kernel_size, stride, padding=0):
    N, C, H, W = input_shape
    K = kernel_size
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    
    X_col = X_col.reshape(N, C, K, K, H_out, W_out)
    
    X = np.zeros((N, C, H, W))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            X[:, :, h_start:h_start+K, w_start:w_start+K] += \
                X_col[:, :, :, :, i, j]
    
    return X


if __name__ == '__main__':
    X = np.arange(1*3*6*6).reshape(1, 3, 6, 6)
    print(X)
    y = to_col(X, 4, 1)
    print(y)
    Y = to_im(y, (1, 3, 6, 6), 4, 1)
    print(Y)