import numpy as np
import pickle

TrainSample = 5000

RunningMode = 'train'


def set_mode(mode):
    RunningMode = mode


def data_monitor(X):
    avg = np.mean(X)
    var = np.var(X)

    print(f"Average: {avg}, Variance: {var}")

    return avg, var


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model