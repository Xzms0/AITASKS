import pickle
import numpy as np


def load_cifar10(path, batch=5, normalize=True):
    def unpickle(file_name):
        with open(path / file_name, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    
    print("Loading CIFAR-10...")
    
    # 加载
    meta = unpickle('batches.meta')
    test_dict = unpickle('test_batch')
    
    X_train_list, y_train_list = [], []
    for i in range(min(batch, 5)):
        d = unpickle(f'data_batch_{i+1}')
        X_train_list.append(d[b'data'])
        y_train_list.append(d[b'labels'])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    idx = np.random.permutation(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    X_test = test_dict[b'data']
    y_test = np.array(test_dict[b'labels'])
    
    # Reshape
    X_train = X_train.reshape(-1, 3, 32, 32).astype(np.float32)
    X_test = X_test.reshape(-1, 3, 32, 32).astype(np.float32)
    
    # 归一化
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    label_list = [name.decode('utf-8') for name in meta[b'label_names']]

    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"y_train {y_train.shape}, y_test {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, label_list