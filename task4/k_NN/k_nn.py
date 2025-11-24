import pickle
from pathlib import Path
import random

import numpy as np

CIFAR_DIR = Path(__file__).absolute().parent.parent / "cifar_10"
METRIC = 'L2'
DATA_BATCH = 5
VAL_SAMPLE = 1000
K = range(1,16)

class kNN:
    def __init__(self, k=8):
        self.k = k
        self.meta_dict = None
        self.test_dict = None
        self.train_dict = []

        self.train_data = None
        self.train_label = None

        self.test_data = None
        self.test_label = None
    

    def _distances(self, train_data, test_data, pieces):
        train_data = np.array(train_data[np.newaxis,:,])
        test_data = np.array(test_data[:,np.newaxis,])

        test_num = test_data.shape[0]
        distances_list = []
        for start in range(0, test_num, pieces):

            end = min(start + pieces, test_num)

            if METRIC == 'L1':
                distances_array = np.array(np.sum(np.abs(test_data[start:end,:,] - train_data), axis=2))
            elif METRIC == 'L2':
                distances_array = np.sqrt(np.sum((test_data[start:end,:,] - train_data)**2, axis=2))

            print(f"-> Got {METRIC} distances_{start//pieces+1} array shaped {distances_array.shape}")
            distances_list.append(distances_array)

        distances = self._to_array(distances_list, msg=f'{METRIC} distances_total', axis=0)
        return distances


    def _to_array(self, data_list: list, msg='an', axis=0):
        array = np.concatenate(data_list, axis=axis)
        print(f"-> Got {msg} array shaped {array.shape}")
        return array
    
    
    def loads(self):
        def unpickle(file_name: str) -> dict:
            with open(CIFAR_DIR / file_name,'rb') as file:
                dict = pickle.load(file,encoding='bytes')

            print(f"-> Loaded {file_name} with {dict.keys()}")
            return dict
        
        print("Start loading cifar-10 batches...")
        self.meta_dict = unpickle('batches.meta')
        self.test_dict = unpickle('test_batch')
        for i in range(DATA_BATCH):
            self.train_dict.append(unpickle(f'data_batch_{i+1}'))

        self.train_data = self._to_array([self.train_dict[i].get(b'data') for i in range(len(self.train_dict))], msg="train_data")
        self.train_label = self._to_array([self.train_dict[i].get(b'labels') for i in range(len(self.train_dict))], msg="train_label")

        self.test_data = self._to_array([self.test_dict.get(b'data')], msg="test_data")
        self.test_label = self._to_array([self.test_dict.get(b'labels')], msg="test_label")


    def train(self, val_sample=None):
        self.train_data = self.train_data
        self.train_label = self.train_label

        if val_sample is not None:
            print(f"\nStart validation with {val_sample} samples...")
            index = list(range(self.train_data.shape[0]))

            random.seed(10)
            random.shuffle(index)
            val_index = index[:val_sample]
            trn_index = index[val_sample:]

            trn_data = self.train_data[trn_index]
            trn_label = self.train_label[trn_index]

            val_data = self.train_data[val_index]
            val_label = self.train_label[val_index]

            print(f"-> Got trn_data array shaped {trn_data.shape}")
            print(f"-> Got trn_label array shaped {trn_label.shape}")
            print(f"-> Got val_data array shaped {val_data.shape}")
            print(f"-> Got val_label array shaped {val_label.shape}")

            best_k = [0, 0]
            for k in K:
                self.k = k
                print(f"\nValidating with K valued {self.k}...")
                val_predict = self.predict(train_data=trn_data, train_label=trn_label, test_data=val_data, pieces=25)
                val_acc = self.accuracy(val_predict, val_label)

                if val_acc > best_k[1]:
                    best_k = [self.k, val_acc]
                print(f"The {self.k} validation got {val_acc*100:.2f}% accuracy.")

            self.k = best_k[0]
            print(f"\nEnd validation with best K valued {self.k}")


    def predict(self, train_data=None, train_label=None, test_data=None, pieces=20):
        if train_data is None:
            train_data = self.train_data
        if test_data is None:
            test_data = self.test_data
        if train_label is None:
            train_label =self.train_label
        
        distances = self._distances(train_data, test_data, pieces=pieces)
        min_indexs = np.argpartition(distances, self.k, axis=1)[:,:self.k]
        predict_k = np.array(train_label[min_indexs])
        predictions = np.array([np.bincount(sample).argmax() for sample in predict_k])
        print(f"-> Got predictions array shaped {predictions.shape}")
        return predictions
    

    def accuracy(self, predict, labels=None):
        if labels is None:
            labels = self.test_label

        acc = np.mean(predict == labels)
        return acc


if __name__ == '__main__':
    knn = kNN()
    knn.loads()
    knn.train(VAL_SAMPLE)
    predict = knn.predict(test_data=knn.test_data[:2000])
    acc = knn.accuracy(predict, labels=knn.test_label[:2000])
    print(f"The final predict got {acc*100:.2f}% accuracy.")


