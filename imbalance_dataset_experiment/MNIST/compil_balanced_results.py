import torch
number_runs = 4
from tqdm import tqdm
from misc.metrics import *
from pre_processing import *
from conditional_density_estimation import *

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

    def train(self, epochs, batch_size,train_samples, train_labels,test_samples, test_labels, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        train_loss_trace = []
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
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    indices.append(__)
                    pbar.set_postfix_str('train_loss = ' + str(round(train_loss, 4)) + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return train_loss_trace, test_loss_trace, indices

class GenerativeClassifierSemiSupervised(torch.nn.Module):
    def __init__(self, samples_dim, labels_dim, structure, prior_probs=None):
        super().__init__()
        self.sample_dim = samples_dim
        self.C = labels_dim
        self.conditional_model = FlowConditionalDensityEstimation(torch.randn(1, samples_dim),
                                                                  torch.ones(1, labels_dim), structure)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.C) / self.C)
        else:
            self.prior_log_probs = torch.log(prior_probs)

    def compute_number_params(self):
        return self.conditional_model.compute_number_params()

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def loss(self, samples, labels):
        return -torch.mean(self.conditional_model.log_prob(samples, labels))

    def log_posterior_prob(self, samples, prior):
        return torch.softmax(self.log_prob(samples) + torch.log(prior.unsqueeze(0)), dim=-1)

    def train(self, epochs, batch_size, train_samples, train_prior_probs, train_labels, test_samples,
              test_prior_probs, test_labels, recording_frequency=1, lr=5e-3,
              weight_decay=5e-5):
        self.conditional_model.initialize_with_EM(torch.cat([train_samples, test_samples], dim=0), 50)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        current_samples = torch.cat([train_samples, test_samples])
        current_labels = torch.cat(
            [train_labels, test_prior_probs.unsqueeze(0).repeat(test_samples.shape[0], 1)])
        dataset = torch.utils.data.TensorDataset(current_samples, current_labels)
        aggregate_loss_trace = []
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
                    aggregate_loss = self.loss(current_samples, current_labels).item()
                    aggregate_loss_trace.append(aggregate_loss)
                    train_loss = self.loss(train_samples, train_labels).item()
                    train_loss_trace.append(train_loss)
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    pbar.set_postfix_str('aggregate_loss = ' + str(round(aggregate_loss, 4))
                                         + '; train_loss = ' + str(round(train_loss, 4))
                                         + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return aggregate_loss_trace, train_loss_trace, test_loss_trace

###Retrieving discriminative results###

list_train_accuracy = []
list_test_accuracy = []
list_train_loss_trace = []
list_test_loss_trace = []
for run in range(number_runs):
    model_disc = torch.load("disc_MNIST_balanced/model_disc_" + str(run) +".pt")
    train_samples, train_prior_probs, train_labels,test_samples, test_prior_probs, test_labels,logit_transform, pca_transform = torch.load("disc_MNIST_balanced/datasets_" + str(run) + ".pt")
    list_train_accuracy.append(compute_accuracy(model_disc.log_prob(train_samples), train_labels))
    list_test_accuracy.append(compute_accuracy(model_disc.log_prob(test_samples), test_labels))
    train_loss_trace = torch.load("disc_MNIST_balanced/train_loss_trace_" + str(run) +".pt")
    test_loss_trace = torch.load("disc_MNIST_balanced/test_loss_trace_" + str(run) +".pt")
    list_train_loss_trace.append(train_loss_trace)
    list_test_loss_trace.append(test_loss_trace)
mean_train_accuracy = torch.tensor(list_train_accuracy).mean()
mean_test_accuracy = torch.tensor(list_test_accuracy).mean()
std_train_accuracy = torch.tensor(list_train_accuracy).std()
std_test_accuracy = torch.tensor(list_test_accuracy).std()
print("mean train accuracy = " + str(mean_train_accuracy) + "+/-" + str(std_train_accuracy))
print("mean test accuracy = " + str(mean_test_accuracy) + "+/-" + str(std_test_accuracy))

mean_train_loss_trace = torch.tensor(list_train_loss_trace).mean(dim = 0)
mean_test_loss_trace = torch.tensor(list_test_loss_trace).mean(dim = 0)
std_train_loss_trace = torch.tensor(list_train_loss_trace).std(dim = 0)
std_test_loss_trace = torch.tensor(list_test_loss_trace).std(dim = 0)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(mean_train_loss_trace, alpha = .6)
ax.fill_between(range(mean_train_loss_trace.shape[0]),mean_train_loss_trace-3*std_train_loss_trace,mean_train_loss_trace+3*std_train_loss_trace, alpha = .6)
ax.plot(mean_test_loss_trace, alpha = .6)
ax.fill_between(range(mean_test_loss_trace.shape[0]),mean_test_loss_trace-3*std_test_loss_trace,mean_test_loss_trace+3*std_test_loss_trace, alpha = .6)

###Retrieving generative results###


list_train_accuracy = []
list_test_accuracy = []
list_train_loss_trace = []
list_test_loss_trace = []
for run in range(number_runs):
    model_disc = torch.load("gen_MNIST_balanced/model_gen_" + str(run) +".pt")
    train_samples, train_prior_probs, train_labels,test_samples, test_prior_probs, test_labels,logit_transform, pca_transform = torch.load("gen_MNIST_balanced/datasets_" + str(run) + ".pt")
    list_train_accuracy.append(compute_accuracy(model_disc.log_prob(train_samples), train_labels))
    list_test_accuracy.append(compute_accuracy(model_disc.log_prob(test_samples), test_labels))
    train_loss_trace = torch.load("gen_MNIST_balanced/train_loss_trace_" + str(run) +".pt")
    test_loss_trace = torch.load("gen_MNIST_balanced/test_loss_trace_" + str(run) +".pt")
    list_train_loss_trace.append(train_loss_trace)
    list_test_loss_trace.append(test_loss_trace)
mean_train_accuracy = torch.tensor(list_train_accuracy).mean()
mean_test_accuracy = torch.tensor(list_test_accuracy).mean()
std_train_accuracy = torch.tensor(list_train_accuracy).std()
std_test_accuracy = torch.tensor(list_test_accuracy).std()
print("mean train accuracy = " + str(mean_train_accuracy) + "+/-" + str(std_train_accuracy))
print("mean test accuracy = " + str(mean_test_accuracy) + "+/-" + str(std_test_accuracy))

mean_train_loss_trace = torch.tensor(list_train_loss_trace).mean(dim = 0)
mean_test_loss_trace = torch.tensor(list_test_loss_trace).mean(dim = 0)
std_train_loss_trace = torch.tensor(list_train_loss_trace).std(dim = 0)
std_test_loss_trace = torch.tensor(list_test_loss_trace).std(dim = 0)

ax = fig.add_subplot(122)
ax.plot(mean_train_loss_trace, alpha = .6)
ax.fill_between(range(mean_train_loss_trace.shape[0]),mean_train_loss_trace-3*std_train_loss_trace,mean_train_loss_trace+3*std_train_loss_trace, alpha = .6)
ax.plot(mean_test_loss_trace, alpha = .6)
ax.fill_between(range(mean_test_loss_trace.shape[0]),mean_test_loss_trace-3*std_test_loss_trace,mean_test_loss_trace+3*std_test_loss_trace, alpha = .6)
plt.show()