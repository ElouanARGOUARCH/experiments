import torch
from classifiers import *
from pre_processing import *
from targets import *

def shuffle(tensor, randperm=None):
    if randperm is None:
        randperm = torch.randperm(tensor.shape[0])
    return tensor[randperm], randperm

logit_transform = logit(alpha = 1e-6)
samples, labels = get_MNIST_dataset(one_hot = True)
n_dim = 63
pca_transform = PCA(samples, n_components=n_dim)
samples, randperm =shuffle(logit_transform.transform(samples))
labels = shuffle(labels, randperm)[0].float()
samples= pca_transform.transform(samples)

sample_list = []
label_list = []
for i in range(2):
    labels_i = labels[labels[:,i]==1][:int(6000*(i+1)/10)]
    samples_i = samples[labels[:, i] == 1][:int(6000 * (i + 1)/10)]
    sample_list.append(samples_i)
    label_list.append(labels_i)
labels = torch.cat(label_list)
samples = torch.cat(sample_list)
labels = torch.argmax(labels, dim = 1)
print(labels.shape)
import matplotlib.pyplot as plt
plt.hist(labels.numpy())
plt.show()
labels = torch.nn.functional.one_hot(labels)
classifier = Classifier(samples, labels, hidden_dimensions= [128,128,128])
classifier.train(500,3000, verbose = True)

prob = torch.sum(torch.exp(classifier.log_prob(torch.randn(10000, n_dim))), dim=0)
print(prob)

