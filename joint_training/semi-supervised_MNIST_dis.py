import torch
from pre_processing import *
from classifiers import *
from targets import *
import numpy as np

def shuffle(tensor, randperm=None):
    if randperm is None:
        randperm = torch.randperm(tensor.shape[0])
    return tensor[randperm], randperm

samples, labels = get_FashionMNIST_dataset(one_hot = True)
pca = PCA(samples, n_components=60, visual = True)
number_to_switch = int(samples.shape[0]/50)
train_samples = samples[:number_to_switch]
train_labels = labels[:number_to_switch]
test_samples = samples[number_to_switch:]
test_labels = labels[number_to_switch:]


train_samples, randperm = shuffle(pca.transform(train_samples))
train_labels,_ = shuffle(train_labels, randperm)
test_samples, randperm = shuffle(pca.transform(test_samples))
test_labels,_ = shuffle(test_labels, randperm)


genclass = Classifier(train_samples, train_labels, [64,64,64,64])
train_accuracy, test_accuracy = genclass.train(200,5000,verbose=True, lr=5e-3, test_samples = test_samples, test_labels = test_labels, trace_accuracy = True)
torch.save(genclass, 'FashionMNIST_models/class_0.pt')
for t in range(20):
    current_labels = torch.nn.functional.one_hot(torch.distributions.Categorical(torch.exp(genclass.log_prob(test_samples))).sample())
    samples_plus = torch.cat([train_samples, test_samples],dim = 0)
    labels_plus = torch.cat([train_labels, current_labels],dim = 0)
    genclass = Classifier(train_samples, train_labels, [64,64,64,64])
    current_train_accuracy, current_test_accuracy = genclass.train(200,5000,verbose=True, lr=5e-3, test_samples = test_samples, test_labels = test_labels, trace_accuracy = True)
    torch.save(genclass, 'FashionMNIST_models/class' + str(t+1)+'.pt')
    train_accuracy += current_train_accuracy
    test_accuracy += current_test_accuracy

with open('train_accuracy.npy', 'wb') as f:
    np.save(f, np.array(train_accuracy))
with open('test_accuracy.npy', 'wb') as f:
    np.save(f, np.array(test_accuracy))

