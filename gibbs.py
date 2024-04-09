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
unlabeled_samples, unlabeled_labels = samples[:20000], labels[:20000].float()
samples, labels = samples[20000:60000], labels[20000:60000].float()

r = range(1, 11)
train_prior_probs = torch.tensor([1 for i in r])
train_prior_probs = train_prior_probs/torch.sum(train_prior_probs)
test_prior_probs = train_prior_probs.flip(dims = [0])
unlabeled_prior_probs = torch.tensor([1 for i in r])
unlabeled_prior_probs = unlabeled_prior_probs/torch.sum(unlabeled_prior_probs)


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

plt.bar(r, torch.sum(unlabeled_labels, dim = 0), color = 'pink', alpha = .4)
plt.bar(r, torch.sum(train_labels, dim = 0), color = 'C1', alpha = .4)
plt.bar(r, torch.sum(test_labels, dim = 0), color = 'green', alpha = .4)
plt.show()

class GenerativeClassifier(torch.nn.Module):
    def __init__(self, samples_dim, labels_dim, structure, prior_probs=None):
        super().__init__()
        self.sample_dim = samples_dim
        self.C = labels_dim
        self.structure = structure
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
        return -torch.mean(torch.sum(self.log_prob(samples)*labels, dim = -1), dim = 0)

    def log_posterior_prob(self, samples, prior):
        log_joint = self.log_prob(samples) + torch.log(prior.unsqueeze(0))
        return log_joint - torch.logsumexp(log_joint, dim = -1, keepdim=True)

    def train(self, epochs, batch_size, train_samples, train_labels, list_test_samples = [], list_test_prior_probs = [],list_test_labels = [],verbose = False, recording_frequency=1, lr=5e-3, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        total_samples = torch.cat([train_samples] + list_test_samples, dim = 0)
        total_labels = torch.cat([train_labels] + [list_test_prior_probs[i].unsqueeze(0).repeat(list_test_samples[i].shape[0],1) for i in range(len(list_test_prior_probs))], dim=0)
        dataset = torch.utils.data.TensorDataset(total_samples, total_labels)
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
                        postfix_str += '; test_loss_' + str(i) + ' = ' + str(round(test_loss, 4))
                    pbar.set_postfix_str(postfix_str)
        self.to(torch.device('cpu'))
        if verbose:
            return train_loss_trace, list_test_loss_trace

    def gibbs(self, T, epochs, batch_size, train_samples, train_labels,list_test_samples = [], list_test_prior_probs = [], list_test_labels = [], recording_frequency = 1, lr = 5e-3, weight_decay = 5e-5):
        self.train(epochs, batch_size, train_samples, train_labels, [],[],[],False,recording_frequency, lr, weight_decay)
        total_samples = torch.cat([train_samples] + list_test_samples, dim = 0)
        print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
        total_labels = [train_labels]
        for i in range(len(list_test_samples)):
            print(compute_accuracy(model_gen.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]),list_test_labels[i]))
            total_labels += [torch.nn.functional.one_hot(torch.distributions.Categorical(torch.exp(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]))).sample(),num_classes=self.C)]
        total_labels = torch.cat(total_labels, dim=0)
        for t in range(T):
            self.conditional_model = FlowConditionalDensityEstimation(torch.randn(1, self.sample_dim),torch.ones(1, self.C), structure)
            self.train(epochs, batch_size, total_samples, total_labels, [],[],[],False,recording_frequency, lr, weight_decay)
            print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
            total_labels = [train_labels]
            for i in range(len(list_test_samples)):
                print(compute_accuracy(model_gen.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]), list_test_labels[i]))
                total_labels += [torch.nn.functional.one_hot(torch.distributions.Categorical(torch.exp(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]))).sample(), num_classes=self.C)]
            total_labels = torch.cat(total_labels, dim=0)

sample_dim = train_samples.shape[-1]
C = train_labels.shape[-1]
structure = [[ConditionalRealNVPLayer, {'hidden_dims': [80, 80, 80]}] for i in range(6)] + [[ConditionalDIFLayer, {'hidden_dims': [32, 32], 'K': 3}] for i in range(1)]
model_gen = GenerativeClassifier(sample_dim,C, structure)
print(model_gen.compute_number_params())
model_gen.gibbs(20,200, int(train_samples.shape[0]/20),train_samples, train_labels, [test_samples], [test_prior_probs],[test_labels])

