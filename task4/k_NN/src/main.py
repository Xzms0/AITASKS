import pickle
from pathlib import Path
import random
import re

import numpy as np
from numpy.typing import NDArray


ROOT_DIR = Path(__file__).absolute().parent.parent
CIFAR_DIR = Path(__file__).absolute().parent.parent.parent / "cifar_10"
ADAPTIVE_BATCH = 10**6
DATA_BATCH = 5

METRIC = 'L1'
VAL_SAMPLE = 5000
TEST_SAMPLE = 10000
TRAIN_SAMPLE = 50000
K = range(1,21)


class Logger:
    def __init__(self, file_name: str, logger_type: str, config: dict={}):
        self.file_name: str = file_name
        self.logger_type: str = logger_type
        self.record: dict = {}
        self.write_config: bool = True
        self.update_record: bool = False

        with open(ROOT_DIR / "data" / self.file_name, 'a', encoding='utf-8'):
            pass
        
        self.METRIC = config.get('metric') if config.get('metric') else METRIC
        self.VAL_SAMPLE = config.get('val_sample') if config.get('val_sample') else VAL_SAMPLE
        self.TEST_SAMPLE = config.get('test_sample') if config.get('test_sample') else TEST_SAMPLE
        self.TRAIN_SAMPLE = config.get('train_sample') if config.get('train_sample') else TRAIN_SAMPLE

        self._reader()


    def _reader(self):
        metric, train_sample, test_sample, val_sample = None, None, None, None

        with open(ROOT_DIR / "data" / self.file_name, 'r', encoding='utf-8') as file:
            for line in file:
                config_1 = re.search(r"METRIC: (L[12]), TRAIN_SAMPLE: ([0-9]{1,})\n", line)
                config_2 = re.search(r"TEST_SAMPLE: ([0-9]{1,}), VAL_SAMPLE: ([0-9]{1,})\n", line)
                if config_1 is not None:
                    metric, train_sample = config_1.group(1), int(config_1.group(2))
                if config_2 is not None:
                    test_sample, val_sample = int(config_2.group(1)), int(config_2.group(2))

                if (metric, train_sample) == (self.METRIC, self.TRAIN_SAMPLE):
                    k_accuracy = re.search(r"K: ([0-9]{1,}), Accuracy: ([0-9.]{1,})%\n", line)

                    if k_accuracy is not None:
                        k, accuracy = int(k_accuracy.group(1)), float(k_accuracy.group(2)) / 100
                        key = {"Val": (val_sample, k),
                               "Test": (test_sample, k)}[self.logger_type]
                        self.record[key] = accuracy
                        
                    if (val_sample, test_sample) == (self.VAL_SAMPLE, self.TEST_SAMPLE): 
                        self.write_config = False

                end_line = re.search(r"[-]{1,}\n", line)
                if end_line is not None:
                    metric, train_sample, test_sample, val_sample = None, None, None, None
                    
        print(f"-> Got {len(self.record)} accuracy items from {self.file_name}")


    def config(self):
        if self.write_config:
            with open(ROOT_DIR / "data" / self.file_name, 'a', encoding='utf-8') as file:
                file.write('-'*36+'\n')
                file.write(f"METRIC: {METRIC}, TRAIN_SAMPLE: {TRAIN_SAMPLE}\nTEST_SAMPLE: {TEST_SAMPLE}, VAL_SAMPLE: {VAL_SAMPLE}\n\n")
            print(f"{self.logger_type}_accuracy record will update...")
            self.write_config = False
            self.update_record = True


    def writer(self, k: int, acc: float, msg="Val"):
        key = {"Val": (VAL_SAMPLE, k),
               "Test": (TEST_SAMPLE, k)}[self.logger_type]
        
        if self.record.get(key):
            acc = self.record.get(key, 0)
            print(f"{msg}_K: {k}, Accuracy: {acc*100:.2f}% - from {self.file_name}")
            if self.update_record:
                with open(ROOT_DIR / "data" / self.file_name, 'a', encoding='utf-8') as file:
                    file.write(f">>> {msg}_K: {k}, Accuracy: {acc*100:.2f}%\n")
        else:
            print(f"{msg}_K: {k}, Accuracy: {acc*100:.2f}%")
            with open(ROOT_DIR / "data" / self.file_name, 'a', encoding='utf-8') as file:
                file.write(f">>> {msg}_K: {k}, Accuracy: {acc*100:.2f}%\n")


class kNN:
    def __init__(self, k=8):
        self.k = k
        self.meta_dict = None
        self.test_dict = None
        self.train_dict = []

        self.train_data: NDArray = np.array([])
        self.train_label: NDArray = np.array([])

        self.test_data: NDArray = np.array([])
        self.test_label: NDArray = np.array([])


    def distances(self, train_data: NDArray | None=None, test_data: NDArray | None=None, pieces=None):
        if train_data is None: 
            train_data = self.train_data

        if test_data is None: 
            test_data = self.test_data

        test_num = test_data.shape[0]
        train_num = train_data.shape[0]

        if pieces is None:
            pieces = ADAPTIVE_BATCH // (train_num * train_data.itemsize)

        distances = np.zeros((test_num, train_num))
        print(f"Start calculating {METRIC} distances array shaped {distances.shape}...")
        for start in range(0, test_num, pieces):
            end = min(start + pieces, test_num)
            process = end / test_num
            process_show = int(process * 24)

            distances_array = np.array([])
            if METRIC == 'L1':
                distances_array = np.array(np.sum(np.abs(test_data[start:end,np.newaxis,] - train_data[np.newaxis,:,]), axis=2))
            elif METRIC == 'L2':
                distances_array = np.sqrt(np.sum((test_data[start:end,np.newaxis,] - train_data[np.newaxis,:,])**2, axis=2))

            #print(f"-> Got {METRIC} distances_{start//pieces+1} array shaped {distances_array.shape}")
            print(f"Process: {process * 100:.2f}% " + "/" * process_show + "-" * (24 - process_show), end='\r')
            distances[start:end,:] = distances_array

        print(f"-> Got {METRIC} distance_total array shaped {distances.shape}")
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
        
        def to_array(data_list: list, msg='an', axis=0) -> NDArray:
            array = np.concatenate(data_list, axis=axis)
            print(f"-> Got {msg} array shaped {array.shape} dtype {array.dtype}")
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
            val_distances = None
            val_logger = Logger('val_accuracy.txt', 'Val')
            for k in K:
                if val_logger.record.get((VAL_SAMPLE, k)):
                    val_acc = val_logger.record.get((VAL_SAMPLE, k), 0)
                else:
                    if val_distances is None:
                        val_distances = self.distances(train_data=trn_data, test_data=val_data)

                    val_predict = self.predict(distances=val_distances, train_label=trn_label, k=k)
                    val_acc = self.accuracy(val_predict, val_label)

                if val_acc > best_k[1]:
                    best_k = [k, val_acc]

                val_logger.config()
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

    test_data = knn.test_data[:TEST_SAMPLE]
    test_label = knn.test_label[:TEST_SAMPLE]

    print(f"Start testing with {TEST_SAMPLE} samples...")
    print(f"-> Got test_data array shaped {test_data.shape}")
    print(f"-> Got test_label array shaped {test_label.shape}")

    best_k = [0, 0]
    test_acc = 0
    distances = None
    test_logger = Logger('test_accuracy.txt', 'Test')
    for k in K:
        if test_logger.record.get((TEST_SAMPLE, k)):
            acc = test_logger.record.get((TEST_SAMPLE, k), 0)
        else:
            if distances is None:
                distances = knn.distances(test_data=test_data)

            predict = knn.predict(distances=distances, k=k)
            acc = knn.accuracy(predict, labels=test_label)
            
        if acc > best_k[1]:
            best_k = [k, acc]
        if k == knn.k:
            test_acc = acc

        test_logger.config()
        test_logger.writer(k, acc, msg='Test')

    test_logger.writer(best_k[0], best_k[1], msg='Best')
    test_logger.writer(knn.k, test_acc, msg='Real')

