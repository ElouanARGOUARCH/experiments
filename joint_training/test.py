with open('train_accuracy.txt','r') as f:
    temp = f.read()
import matplotlib.pyplot as plt
plt.plot(list(temp))
plt.show()