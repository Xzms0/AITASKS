import pickle
from pathlib import Path
import random

import numpy as np

ROOT_DIR = Path(__file__).absolute().parent
CIFAR_DIR = Path(__file__).absolute().parent.parent / "cifar_10"
ADAPTIVE_BATCH = 10**6

METRIC = 'L2'
DATA_BATCH = 5
VAL_SAMPLE = 5000
TEST_SAMPLE = 10000
TRAIN_SAMPLE = 50000
K = range(1,21)


class Logger:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.record = {}

        with open(ROOT_DIR / self.file_name, 'a', encoding='utf-8') as file:
            file.write('-'*36+'\n')
            file.write(f"METRIC: {METRIC}, TRAIN_SAMPLE: {TRAIN_SAMPLE}\nTEST_SAMPLE: {TEST_SAMPLE}, VAL_SAMPLE: {VAL_SAMPLE}\n\n")
    
    def reader(self):
        pass


    def writer(self, k: int, acc: float, msg="Val"):
        print(f"{msg}_K: {k}, Accuracy: {acc*100:.2f}%")
        with open(ROOT_DIR / self.file_name, 'a', encoding='utf-8') as file:
            file.write(f">>> {msg}_K: {k}, Accuracy: {acc*100:.2f}%\n")


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
    
    
    def distances(self, train_data=None, test_data=None, pieces=None):
        if train_data is None: 
            train_data = self.train_data
        if test_data is None: 
            test_data = self.test_data

        test_num = test_data.shape[0]
        train_num = train_data.shape[0]

        if pieces is None:
            pieces = ADAPTIVE_BATCH // (train_num * train_data.itemsize)

        distances = np.zeros((test_num, train_num))
        for start in range(0, test_num, pieces):
            end = min(start + pieces, test_num)
            process = end / test_num
            process_show = int(process * 24)

            if METRIC == 'L1':
                distances_array = np.array(np.sum(np.abs(test_data[start:end,np.newaxis,] - train_data[np.newaxis,:,]), axis=2))
            elif METRIC == 'L2':
                distances_array = np.sqrt(np.sum((test_data[start:end,np.newaxis,] - train_data[np.newaxis,:,])**2, axis=2))

            print(f"-> Got {METRIC} distances_{start//pieces+1} array shaped {distances_array.shape}")
            print(f"Process: {process * 100:.2f}% " + "/" * process_show + "-" * (24 - process_show), end='\r')
            distances[start:end,:] = distances_array

        print(f"-> Got distance_total array shaped {distances.shape}")
        return distances


    def normalize(self):
        if self.train_data is not None:
            self.train_data = self.train_data.astype(np.float32) / 255.0
        if self.test_data is not None:
            self.test_data = self.test_data.astype(np.float32) / 255.0

        print("Data is normalized to float32...")
    

    def loads(self):
        def unpickle(file_name: str) -> dict:
            with open(CIFAR_DIR / file_name,'rb') as file:
                dict = pickle.load(file,encoding='bytes')

            print(f"-> Loaded {file_name} with {dict.keys()}")
            return dict
        
        def to_array(data_list: list, msg='an', axis=0):
            array = np.concatenate(data_list, axis=axis)
            print(f"-> Got {msg} array shaped {array.shape}")
            return array
        
        print("Start loading cifar-10 batches...")
        self.meta_dict = unpickle('batches.meta')
        self.test_dict = unpickle('test_batch')
        for i in range(DATA_BATCH):
            self.train_dict.append(unpickle(f'data_batch_{i+1}'))

        self.train_data = to_array([self.train_dict[i].get(b'data') for i in range(len(self.train_dict))], msg="train_data")
        self.train_label = to_array([self.train_dict[i].get(b'labels') for i in range(len(self.train_dict))], msg="train_label")

        self.test_data = to_array([self.test_dict.get(b'data')], msg="test_data")
        self.test_label = to_array([self.test_dict.get(b'labels')], msg="test_label")


    def train(self, val_sample=None):
        self.train_data = self.train_data[:TRAIN_SAMPLE]
        self.train_label = self.train_label[:TRAIN_SAMPLE]

        if val_sample is not None:
            print(f"Start validation with {val_sample} samples...")
            index = list(range(self.train_data.shape[0]))
            val_logger = Logger('val_accuracy.txt')

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
            val_distances = self.distances(train_data=trn_data, test_data=val_data)
            for k in K:
                val_predict = self.predict(distances=val_distances, train_label=trn_label, k=k)
                val_acc = self.accuracy(val_predict, val_label)

                if val_acc > best_k[1]:
                    best_k = [k, val_acc]

                val_logger.writer(k, val_acc, msg='Val')

            self.k = best_k[0]
            val_logger.writer(best_k[0], best_k[1], msg='Best')


    def predict(self, distances, train_label=None, k=None):
        if train_label is None:
            train_label =self.train_label
        if k is None: 
            k = self.k
        
        min_indexs = np.argpartition(distances, k, axis=1)[:,:k]
        predict_k = np.array(train_label[min_indexs])
        predict = np.array([np.bincount(sample).argmax() for sample in predict_k])

        #print(f"-> Got predict array shaped {predict.shape}")
        return predict
    

    def accuracy(self, predict, labels=None):
        if labels is None:
            labels = self.test_label

        acc = np.mean(predict == labels)
        return acc


if __name__ == '__main__':
    knn = kNN()
    knn.loads()
    knn.normalize()
    knn.train(VAL_SAMPLE)

    best_k = [0, 0]
    test_acc = 0
    test_logger = Logger('test_accuracy.txt')
    distances = knn.distances(test_data=knn.test_data[:TEST_SAMPLE])
    for k in K:
        predict = knn.predict(distances=distances, k=k)
        acc = knn.accuracy(predict, labels=knn.test_label[:TEST_SAMPLE])

        if acc > best_k[1]:
            best_k = [k, acc]
        if k == knn.k:
            test_acc = acc

        test_logger.writer(k, acc, msg='Test')

    test_logger.writer(best_k[0], best_k[1], msg='Best')
    test_logger.writer(knn.k, test_acc, msg='Val')


