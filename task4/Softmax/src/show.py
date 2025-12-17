import numpy as np
from main import ROOT_DIR
import matplotlib.pyplot as plt

weight_name = None
with open(ROOT_DIR / 'data' / 'best_weight.txt', 'r') as f:
    weight_name = f.readline()

weights = np.load(ROOT_DIR / 'data' / f'{weight_name}.npy')
weights = weights[:-1]
weights = weights.transpose()
weights = weights.reshape(10, 3, 32, 32)

w_min, w_max = np.min(weights), np.max(weights)
weights = weights.transpose((0, 2, 3, 1))

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    wimg = 255.0 * (weights[i, :, :, :].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()