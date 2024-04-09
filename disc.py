import torch
from pre_processing import *
from targets import *
from tqdm import tqdm
from misc.metrics import *
from conditional_density_estimation import *
from classifiers import *

def shuffle(tensor, randperm=None):
    if randperm is None:
        randperm = torch.randperm(tensor.shape[0])
    return tensor[randperm], randperm


logit_transform = logit(alpha = 1e-6)
samples, labels = get_MNIST_dataset(one_hot = True)
samples, randperm = shuffle(logit_transform.transform(samples))
labels,_ = shuffle(labels, randperm)
pca_transform = PCA(samples, n_components=100)
samples = pca_transform.transform(samples)
num_samples = torch.sum(labels, dim = 0)
unlabeled_samples, unlabeled_labels = samples[:20000], labels[:20000].float()
samples, labels = samples[20000:60000], labels[20000:60000].float()

r = range(1, 11)
train_prior_probs = torch.tensor([1 for i in r])*num_samples
train_prior_probs = train_prior_probs/torch.sum(train_prior_probs)
test_prior_probs = train_prior_probs.flip(dims = [0])
unlabeled_prior_probs = torch.tensor([1 for i in r])*num_samples
unlabeled_prior_probs = unlabeled_prior_probs/torch.sum(unlabeled_prior_probs)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.bar(r, train_prior_probs, color = 'pink', alpha = .4)
ax.bar(r, unlabeled_prior_probs, color = 'C1', alpha = .4)
ax.bar(r, test_prior_probs, color = 'green', alpha = .4)

train_samples, train_labels = [],[]
test_samples, test_labels = [],[]
for label in range(labels.shape[-1]):
    current_samples = samples[labels[:,label] == 1]
    current_labels = labels[labels[:,label] == 1]
    for_train = current_samples.shape[0]*5*train_prior_probs[label]
    train_samples.append(current_samples[:int(for_train)])
    test_samples.append(current_samples[int(for_train):])
    train_labels.append(current_labels[:int(for_train)])
    test_labels.append(current_labels[int(for_train):])
train_samples, train_labels = torch.cat(train_samples),torch.cat(train_labels)
test_samples, test_labels = torch.cat(test_samples),torch.cat(test_labels)

ax = fig.add_subplot(122)
ax.bar(r, torch.sum(unlabeled_labels, dim = 0), color = 'pink', alpha = .4)
ax.bar(r, torch.sum(train_labels, dim = 0), color = 'C1', alpha = .4)
ax.bar(r, torch.sum(test_labels, dim = 0), color = 'green', alpha = .4)
plt.show()

class Classifier(torch.nn.Module):
    def __init__(self, sample_dim, C, hidden_dimensions=[]):
        super().__init__()
        assert samples.shape[0] == labels.shape[0], 'number of samples does not match number of samples'
        self.sample_dim = sample_dim
        self.C = C
        self.network_dimensions = [self.sample_dim] + hidden_dimensions + [self.C]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.Tanh(), ])
        self.f = torch.nn.Sequential(*network)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_prob(self, samples):
        temp = self.f.forward(samples)
        return temp - torch.logsumexp(temp, dim=-1, keepdim=True)

    def loss(self, samples, labels):
        return -torch.mean(self.log_prob(samples) * labels)

    def train(self, epochs, batch_size,train_samples, train_labels, unlabeled_samples, unlabeled_labels,
              test_samples, test_labels, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        train_loss_trace = []
        unlabeled_loss_trace = []
        test_loss_trace = []
        indices = []
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    train_loss = self.loss(train_samples, train_labels).item()
                    train_loss_trace.append(train_loss)
                    unlabeled_loss = self.loss(unlabeled_samples, unlabeled_labels).item()
                    unlabeled_loss_trace.append(unlabeled_loss)
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    indices.append(__)
                    pbar.set_postfix_str('train_loss = ' + str(round(train_loss, 4)) + '; unlabeled_loss = ' + str(round(unlabeled_loss, 4)) + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices

model_disc = Classifier(train_samples.shape[-1], train_labels.shape[-1],[256,256,256,256])
print(model_disc.compute_number_params())
train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices = model_disc.train(1000,2000,train_samples,train_labels,unlabeled_samples,unlabeled_labels, test_samples, test_labels, recording_frequency=20)
plt.plot(train_loss_trace)
plt.plot(unlabeled_loss_trace)
plt.plot(test_loss_trace)
plt.show()
print(compute_accuracy(model_disc.log_prob(train_samples), train_labels))
print(compute_accuracy(model_disc.log_prob(unlabeled_samples), unlabeled_labels))
print(compute_accuracy(model_disc.log_prob(test_samples), test_labels))