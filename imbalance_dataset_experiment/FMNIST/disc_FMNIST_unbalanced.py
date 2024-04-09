import torch
from pre_processing import *
from targets import *
from tqdm import tqdm
from classifiers import *

class Classifier(torch.nn.Module):
    def __init__(self, sample_dim, C, hidden_dimensions=[]):
        super().__init__()
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

    def train(self, epochs, batch_size,train_samples, train_labels,list_test_samples = [], list_test_labels = [],verbose = False, recording_frequency = 1, lr=5e-4, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        if verbose:
            train_loss_trace = []
            list_test_loss_trace = [[] for i in range(len(list_test_samples))]
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0 and verbose:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    train_loss = self.loss(train_samples, train_labels).item()
                    train_loss_trace.append(train_loss)
                    postfix_str = 'device = ' + str(
                        device) + '; train_loss = ' + str(round(train_loss, 4))
                    for i in range(len(list_test_samples)):
                        test_loss = self.loss(list_test_samples[i], list_test_labels[i]).item()
                        list_test_loss_trace[i].append(test_loss)
                        postfix_str += '; test_loss_'+ str(i) +' = ' + str(round(test_loss, 4))
                    pbar.set_postfix_str(postfix_str)
        self.to(torch.device('cpu'))
        if verbose:
            return train_loss_trace, list_test_loss_trace

number_runs = 5
for run in range(number_runs):
    logit_transform = logit(alpha = 1e-6)
    samples, labels = get_FashionMNIST_dataset(one_hot = True)
    samples, randperm = shuffle(logit_transform.transform(samples))
    labels,_ = shuffle(labels, randperm)
    labels = labels.float()
    pca_transform = PCA(samples, n_components=100)
    samples = pca_transform.transform(samples)
    num_samples = torch.sum(labels, dim = 0)
    r = range(0, 10)
    train_prior_probs = torch.tensor([i+1 for i in r])*num_samples
    train_prior_probs = train_prior_probs/torch.sum(train_prior_probs)
    test_prior_probs = torch.tensor([10-i for i in r])*num_samples
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
    datasets = (train_samples, train_prior_probs, train_labels, [test_samples], [test_prior_probs], [test_labels], logit_transform, pca_transform)
    torch.save(datasets,"disc_FMNIST_unbalanced/datasets_" + str(run) + ".pt")


    model_disc = Classifier(train_samples.shape[-1], train_labels.shape[-1], [256, 256, 256,256])
    print(model_disc.compute_number_params())
    model_disc.train(800,int(70000/20),train_samples,train_labels)
    torch.save(model_disc,"disc_FMNIST_unbalanced/model_disc_" +str(run) + ".pt")
