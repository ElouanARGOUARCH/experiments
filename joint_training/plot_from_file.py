import numpy as np
import matplotlib.pyplot as plt

total_train_accuracy = np.load('train_accuracy.npy')
total_test_accuracy = np.load('test_accuracy.npy')

plt.plot(total_train_accuracy, color = 'red', label = 'train')
plt.plot(total_test_accuracy,color = 'green',label = 'test')
plt.legend()
plt.show()
