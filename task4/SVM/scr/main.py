import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


ROOT_DIR = Path(__file__).absolute().parent.parent
CIFAR_DIR = Path(__file__).absolute().parent.parent.parent / "cifar_10"
DATA_BATCH = 5

TEST_SAMPLE = 10000
TRAIN_SAMPLE = 50000


class SVM:
    def __init__(self):
        self.delta = 1.0
        self.reg = 1e-4 #1e-4
        self.std = 1e-3
        self.step = 3e-3 #3e-3

        try:
            weight_name = None
            with open(ROOT_DIR / 'data' / 'best_weight.txt', 'r') as f:
                weight_name = f.readline()

            self.weights = np.load(ROOT_DIR / 'data' / f'{weight_name}.npy')
        except:
            self.weights = np.array([])

        self.train_data: NDArray = np.array([])
        self.train_label: NDArray = np.array([])

        self.test_data: NDArray = np.array([])
        self.test_label: NDArray = np.array([])

        self.label_list = []

        self.loads()
        self.normalize()


    def normalize(self):
        if self.train_data is not None:
            self.train_data = self.train_data.astype(np.float64) / 255.0
            #self.train_data = (self.train_data - np.mean(self.train_data)) #/ 255.0
        if self.test_data is not None:
            self.test_data = self.test_data.astype(np.float64) / 255.0
            #self.test_data = (self.test_data - np.mean(self.test_data)) #/ 255.0

        print("Data is normalized to float64...")


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
        meta_dict = unpickle('batches.meta')
        test_dict = unpickle('test_batch')
        train_dict = []
        for i in range(DATA_BATCH):
            train_dict.append(unpickle(f'data_batch_{i+1}'))

        self.train_data = to_array([train_dict[i].get(b'data') for i in range(len(train_dict))], msg="train_data")
        self.train_label = to_array([train_dict[i].get(b'labels') for i in range(len(train_dict))], msg="train_label")

        self.test_data = to_array([test_dict.get(b'data')], msg="test_data")
        self.test_label = to_array([test_dict.get(b'labels')], msg="test_label")

        self.label_list = meta_dict[b'label_names']

    def scores(self, data: NDArray, weights=None):
        if weights is None:
            weights = self.weights

        scores = data.dot(weights)
        return scores
    

    def loss(self, scores: NDArray, labels, weights=None):
        if weights is None:
            weights = self.weights

        sample = scores.shape[0]
        correct_scores = scores[range(sample), labels]

        margins = np.array(np.maximum(0, scores - correct_scores[:, np.newaxis] + self.delta))
        margins[range(sample), labels] = 0

        data_loss = np.sum(margins) / sample
        reg_loss = np.sum(weights ** 2) * self.reg

        loss = data_loss + reg_loss
        return loss


    def gradient_analysis(self, weights):
        scores = self.scores(self.train_data, weights)
        sample = scores.shape[0]
        correct_scores = scores[range(sample), self.train_label]

        margins = np.array(np.maximum(0, scores - correct_scores[:, np.newaxis] + self.delta))
        margins[range(sample), self.train_label] = 0

        binary = (margins > 0).astype(np.float32)
        activation = np.sum(binary, axis=1)

        binary[range(sample), self.train_label] = -activation

        data_grad = self.train_data.transpose().dot(binary) / sample
        reg_grad = weights * self.reg

        grad = data_grad + reg_grad
        return grad

    def gradient_value(self, weights):
        scores = self.scores(self.train_data, weights)
        fx = self.loss(scores, self.train_label, weights)
        grad = np.zeros(weights.shape)
        h = 0.00001

        for ix, old_value in np.ndenumerate(weights):

            old_value = weights[ix]
            weights[ix] = old_value + h

            scores = self.scores(self.train_data, weights)
            fxh = self.loss(scores, self.train_label, weights)
            weights[ix] = old_value

            grad[ix] = (fxh - fx) / h 

            show = int(24 * (ix[0]+1) / 3073)
            print(f"Process: {(ix[0]+1) / 3073 * 100:.2f}% " + "/" + "/" * show + "-" * (24 - show) + "/", end='\r')

        return grad
    

    def trick(self):
        trick = np.ones((self.train_data.shape[0],1))
        self.train_data = np.concatenate((self.train_data, trick), axis=1)

        trick = np.ones((self.test_data.shape[0],1))
        self.test_data = np.concatenate((self.test_data, trick), axis=1)

    
    def train(self):
        self.train_data = self.train_data[:TRAIN_SAMPLE]
        self.train_label = self.train_label[:TRAIN_SAMPLE]

        try:
            weight_name = None
            with open(ROOT_DIR / 'data' / 'best_weight.txt', 'r') as f:
                weight_name = f.readline()

            weights = np.load(ROOT_DIR / 'data' / f'{weight_name}.npy')
        except:
            weights = np.random.randn(3073, 10) * self.std
            weights[3072, :] = 0

        best_weights = weights.copy()
        best_loss = float("inf")

        scores = self.scores(self.train_data, weights=weights)
        original_loss = self.loss(scores, self.train_label, weights=weights)
        print(f"The original loss was {original_loss}")

        times = 100
        for i in range(times):
            grad = self.gradient_analysis(weights)
            weights = weights - grad * self.step

            scores = self.scores(self.train_data, weights=weights)
            loss = self.loss(scores, self.train_label, weights=weights)
            
            if loss < best_loss:
                best_weights = weights
                best_loss = loss
            
            print(f"In attempt {i+1}/{times} the loss was {loss}", end='\r')
        
        print()
        self.weights = best_weights
        np.save(ROOT_DIR / 'data' / f'{best_loss}.npy', self.weights)
        with open(ROOT_DIR / 'data' / 'best_weight.txt', 'w') as f:
            f.write(f'{best_loss}')


    def predict(self, scores):
        predict = scores.argmax(axis=1)
        return predict


    def accuracy(self, predict, labels):
        accuracy = np.mean(predict == labels)
        return float(accuracy)


if __name__ == '__main__':
    svm = SVM()
    svm.trick()
    
    #svm.train()
    scores = svm.scores(svm.test_data)
    loss = svm.loss(scores, svm.test_label)
    predict = svm.predict(scores)
    accuracy = svm.accuracy(predict, svm.test_label)

    print(f"\nThe loss was {loss}\nThe accuracy was {accuracy*100:.2f}%\n")
