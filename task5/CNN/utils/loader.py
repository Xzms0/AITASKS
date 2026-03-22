import pickle
from pathlib import Path

import numpy as np


def loader(path, batch):
    def unpickle(file_name: str) -> dict:
            with open(path / file_name,'rb') as file:
                dict = pickle.load(file,encoding='bytes')

            print(f"-> Loaded {file_name} with {dict.keys()}")
            return dict
        
    def to_array(data_list: list, msg='an', axis=0):
        array = np.concatenate(data_list, axis=axis)
        print(f"-> Got {msg} array shaped {array.shape} dtype {array.dtype}")
        return array

    print("Start loading cifar-10 batches...")
    meta_dict = unpickle('batches.meta')
    test_dict = unpickle('test_batch')
    train_dict = []
    for i in range(batch):
        train_dict.append(unpickle(f'data_batch_{i+1}'))

    train_data = to_array([train_dict[i].get(b'data') for i in range(len(train_dict))], msg="train_data")
    train_label = to_array([train_dict[i].get(b'labels') for i in range(len(train_dict))], msg="train_label")

    test_data = to_array([test_dict.get(b'data')], msg="test_data")
    test_label = to_array([test_dict.get(b'labels')], msg="test_label")

    label_list = meta_dict[b'label_names']

    return train_data, train_label, test_data, test_label, label_list