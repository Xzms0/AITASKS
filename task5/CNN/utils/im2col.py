import numpy as np

def to_col(X, size, stride):
    '''
    X: (N, in_chanels, height, width)
    return: (N, in_chanels, size*size, H_out*W_out)
    '''
    N, C, H, W = X.shape
    H_out = (X.shape[2] - size) // stride + 1
    W_out = (X.shape[3] - size) // stride + 1

    cols= np.zeros((N, C, size*size, H_out*W_out))
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + size

            w_start = j * stride
            w_end = w_start + size

            window = X[:, :, h_start:h_end, w_start:w_end]
            col = window.reshape((N, C, -1))
            cols[:, :, :, i*W_out+j] = col

    print(cols.shape)
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
    
    print(X.shape)
    return X


if __name__ == '__main__':
    X = np.arange(1*3*6*6).reshape(1, 3, 6, 6)
    print(X)
    y = to_col(X, 4, 1)
    print(y)
    Y = to_im(y, (1, 3, 6, 6), 4, 1)
    print(Y)