import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


ROOT_DIR = Path(__file__).absolute().parent
CIFAR_DIR = Path(__file__).absolute().parent.parent / "cifar_10"
DATA_BATCH = 5

TEST_SAMPLE = 10000
TRAIN_SAMPLE = 50000

np.random.seed(13)

class NeuralNetwork:
    def __init__(self, reg=5e-6, std=1e-4, step=2e-2):
        self.train_data: NDArray = np.array([])
        self.train_label: NDArray = np.array([])

        self.test_data: NDArray = np.array([])
        self.test_label: NDArray = np.array([])

        self.label_list = []

        self.params = {}
        #W1: 3072*100 b1: 1*100
        #W2: 100*10 b2: 1*10
        self.reg = reg
        self.std = std
        self.step = step

        self.loads()
        self.preprocess()


    def preprocess(self):
        '''
        if self.train_data is not None:
            self.train_data = self.train_data.astype(np.float64) / 255.0
            self.train_data = (self.train_data - np.mean(self.train_data, axis=0)) / (np.std(self.train_data, axis=0) + 1e-8)
        if self.test_data is not None:
            self.test_data = self.test_data.astype(np.float64) / 255.0
            self.test_data = (self.test_data - np.mean(self.test_data, axis=0)) / (np.std(self.test_data, axis=0) + 1e-8)
        '''
        self.train_data = self.train_data.astype(np.float64)
        self.test_data = self.test_data.astype(np.float64)
        mean = np.mean(self.train_data, axis=0)
        self.train_data -= mean
        self.test_data -= mean

        print("Data is preprocessed...")


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


    def relu(self, x):
        return np.maximum(0, x)


    def relu_backward(self, dout, x):
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx


    def loss(self, data, labels=None, params={}):
        if not params:
            params = self.params

        hidden_layer: NDArray = self.relu(data.dot(params['W1']) + params['b1'])
        scores: NDArray = hidden_layer.dot(params['W2']) + params['b2']

        if labels is None:
            return scores

        sample = scores.shape[0]
        
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        data_loss = np.mean(-np.log(probs[range(sample), labels] + 1e-8))  # 加小常数避免log(0)
        reg_loss = 0.5 * self.reg * (np.sum(params['W1'] ** 2) + np.sum(params['W2'] ** 2))
        loss = data_loss + reg_loss

        grads = {}

        dscores = probs.copy()
        dscores[range(sample), labels] -= 1
        dscores /= sample

        grads['W2'] = np.dot(hidden_layer.T, dscores) + self.reg * params['W2']
        grads['b2'] = np.sum(dscores, axis=0)

        dhidden = np.dot(dscores, params['W2'].T)
        dhidden_relu = self.relu_backward(dhidden, hidden_layer)
        
        grads['W1'] = np.dot(data.T, dhidden_relu) + self.reg * params['W1']
        grads['b1'] = np.sum(dhidden_relu, axis=0)

        return loss, grads
    

    def train(self, iter_times=100, batch_size=500):

        self.train_data = self.train_data[:TRAIN_SAMPLE]
        self.train_label = self.train_label[:TRAIN_SAMPLE]

        self.params['W1'] = self.std * np.random.randn(3072, 100)
        self.params['b1'] = np.zeros(100)
        self.params['W2'] = self.std * np.random.randn(100, 10)
        self.params['b2'] = np.zeros(10)

        for key, value in self.params.items():
            print(f"{key} shape: {value.shape}")

        loss_history = []

        for it in range(iter_times):
            mask = np.random.choice(TRAIN_SAMPLE, batch_size, replace=False)
            data_batch = self.train_data[mask]
            label_batch = self.train_label[mask]

            loss, grads = self.loss(data_batch, labels=label_batch, params=self.params)
            #loss, grads = self.loss(self.train_data, labels=self.train_label, params=self.params)

            self.params['W1'] -= self.step * grads['W1']
            self.params['b1'] -= self.step * grads['b1']
            self.params['W2'] -= self.step * grads['W2']
            self.params['b2'] -= self.step * grads['b2']

            if it % 100 == 0 and it != 0:
                self.step *= 0.95

            loss_history.append(loss)
            print(f"Iteration {it+1}/{iter_times}: loss = {loss:.4f}", end='\r')

        print(f"\nFinal loss: {loss_history[-1]:.4f}")
        return loss_history


    def predict(self, data):
        scores = self.loss(data, params=self.params)
        predict = scores.argmax(axis=1)
        return predict


    def accuracy(self, predict, labels):
        accuracy = np.mean(predict == labels)
        return float(accuracy)


def main():
    two_layer = NeuralNetwork(reg=5e-6, std=1e-4, step=1e-3)
    loss_history = two_layer.train(iter_times=10000, batch_size=500)
    predict = two_layer.predict(two_layer.train_data)
    accuracy = two_layer.accuracy(predict, two_layer.train_label)
    print(f"\nTrain Accuracy: {accuracy*100:.2f}%")
    predict = two_layer.predict(two_layer.test_data)
    accuracy = two_layer.accuracy(predict, two_layer.test_label)
    print(f"Test Accuracy: {accuracy*100:.2f}%")


if __name__ == '__main__':
    main()    