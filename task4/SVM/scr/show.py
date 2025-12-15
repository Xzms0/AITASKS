import numpy as np
from main import SVM, ROOT_DIR
import matplotlib.pyplot as plt

score = np.array([[10, 12], [5, 6], [10, 7]])

print(score.argmax(axis=1).shape)

o = np.zeros((10000,3072))
x = np.ones((10000,1))
#np.save("x.npy", x)

print(np.concatenate((o,x), axis=1).shape)

show = SVM()
show.loads()

image = show.train_data[40]
label = int(show.train_label[40])
print(show.label_list[label])

print(image.shape)

image = image.reshape(3, 32, 32)
image = image.transpose((1, 2, 0))
plt.imshow(image)
plt.show()
'''
weights = np.load(ROOT_DIR / 'data' / '7.npy')
weights = (weights[:3072] + 0.017) * 30
print(np.mean(weights))
weights = weights.transpose()
weights = weights.reshape(3, 320, 32)
weights = weights.transpose((2, 1, 0))
plt.xticks([32 * i - 1 for i in range(11)])
plt.imshow(weights)
plt.show()'''