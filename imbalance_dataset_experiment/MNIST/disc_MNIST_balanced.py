import torch
from pre_processing import *
from targets import *
from tqdm import tqdm
from misc.metrics import *
from conditional_density_estimation import *
from classifiers import *
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

    def train(self, epochs, batch_size,train_samples, train_labels,test_samples, test_labels, recording_frequency = 1, lr=5e-4, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        train_loss_trace = []
        test_loss_trace = []
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
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    pbar.set_postfix_str('train_loss = ' + str(round(train_loss, 4)) + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return train_loss_trace, test_loss_trace

number_runs = 10
for run in range(number_runs):
    logit_transform = logit(alpha = 1e-6)
    samples, labels = get_MNIST_dataset(one_hot = True)
    samples, randperm = shuffle(logit_transform.transform(samples))
    labels,_ = shuffle(labels, randperm)
    pca_transform = PCA(samples, n_components=100)
    samples = pca_transform.transform(samples)
    num_samples = torch.sum(labels, dim = 0)
    samples, labels = samples[20000:60000], labels[20000:60000].float()

    r = range(0, 10)
    train_prior_probs = torch.tensor([1 for i in r])*num_samples
    train_prior_probs = train_prior_probs/torch.sum(train_prior_probs)
    test_prior_probs = torch.tensor([1 for i in r])*num_samples
    test_prior_probs = test_prior_probs/torch.sum(test_prior_probs)
    train_samples, train_labels = [],[]
    test_samples, test_labels = [],[]
    for label in range(labels.shape[-1]):
        current_samples = samples[labels[:,label] == 1]
        current_labels = labels[labels[:,label] == 1]
        for_train = current_samples.shape[0]*train_prior_probs[label]/(train_prior_probs[label] + test_prior_probs[label])
        train_samples.append(current_samples[:int(for_train)])
        test_samples.append(current_samples[int(for_train):])
        train_labels.append(current_labels[:int(for_train)])
        test_labels.append(current_labels[int(for_train):])
    train_samples, train_labels = torch.cat(train_samples),torch.cat(train_labels)
    test_samples, test_labels = torch.cat(test_samples),torch.cat(test_labels)
    datasets = (train_samples, train_prior_probs, train_labels, test_samples, test_prior_probs, test_labels, logit_transform, pca_transform)
    torch.save(datasets,"disc_MNIST_balanced/datasets_" + str(run) + ".pt")

    model_disc = Classifier(train_samples.shape[-1], train_labels.shape[-1], [256, 256, 256])
    train_loss_trace, test_loss_trace = model_disc.train(600,int(train_samples.shape[0]/20),train_samples,train_labels, test_samples, test_labels, recording_frequency=10)
    torch.save(train_loss_trace, "disc_MNIST_balanced/train_loss_trace_" + str(run) + ".pt")
    torch.save(test_loss_trace, "disc_MNIST_balanced/test_loss_trace_" + str(run) + ".pt")
    torch.save(model_disc,"disc_MNIST_balanced/model_disc_" +str(run) + ".pt")
