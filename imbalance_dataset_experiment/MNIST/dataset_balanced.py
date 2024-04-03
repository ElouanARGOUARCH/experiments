from pre_processing import *
from targets import *
from conditional_density_estimation import *

logit_transform = logit(alpha=1e-6)
samples, labels = get_MNIST_dataset(one_hot=True)
samples, randperm = shuffle(logit_transform.transform(samples))
labels, _ = shuffle(labels, randperm)
pca_transform = PCA(samples, n_components=100)
samples = pca_transform.transform(samples)
num_samples = torch.sum(labels, dim=0)
samples, labels = samples[20000:60000], labels[20000:60000].float()

r = range(0, 10)
train_prior_probs = torch.tensor([1 for i in r]) * num_samples
train_prior_probs = train_prior_probs / torch.sum(train_prior_probs)
test_prior_probs = torch.tensor([1 for i in r]) * num_samples
test_prior_probs = test_prior_probs / torch.sum(test_prior_probs)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.bar(r, train_prior_probs, color = 'C1', alpha = .4)
ax.bar(r, test_prior_probs, color = 'green', alpha = .4)

train_samples, train_labels = [], []
test_samples, test_labels = [], []
for label in range(labels.shape[-1]):
    current_samples = samples[labels[:, label] == 1]
    current_labels = labels[labels[:, label] == 1]
    for_train = current_samples.shape[0] * train_prior_probs[label] / (
                train_prior_probs[label] + test_prior_probs[label])
    train_samples.append(current_samples[:int(for_train)])
    test_samples.append(current_samples[int(for_train):])
    train_labels.append(current_labels[:int(for_train)])
    test_labels.append(current_labels[int(for_train):])
train_samples, train_labels = torch.cat(train_samples), torch.cat(train_labels)
test_samples, test_labels = torch.cat(test_samples), torch.cat(test_labels)

ax = fig.add_subplot(122)
ax.bar(r, torch.sum(train_labels, dim = 0), color = 'C1', alpha = .4)
ax.bar(r, torch.sum(test_labels, dim = 0), color = 'green', alpha = .4)
plt.show()