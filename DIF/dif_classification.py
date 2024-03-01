from conditional_density_estimation import *
from pre_processing import *
from targets import *
from misc import *

logit_transform = logit(alpha = 1e-6)
samples, labels = get_MNIST_dataset(one_hot = True)
n_dim = 60
pca_transform = PCA(samples, n_components=n_dim)
samples, randperm =shuffle(logit_transform.transform(samples))
labels = shuffle(labels, randperm)[0].float()
samples= pca_transform.transform(samples)

n_samples = samples.shape[0]/10
print(n_samples)

train_samples = samples[n_samples - n_samples/2:]
test_samples = samples[n_samples - 3500:n_samples - 1750]
unlabeled_samples = samples[n_samples - 1750:n_samples]
train_labels = labels[n_samples - 3500:]
test_labels = labels[n_samples - 3500:n_samples - 1750]
unlabeled_labels = labels[n_samples - 1750:n_samples]

print(train_samples.shape,test_samples.shape,unlabeled_samples.shape,train_labels.shape,test_labels.shape,unlabeled_labels.shape)

class GenerativeClassifierSemiSupervised(torch.nn.Module):
    def __init__(self, samples, labels, structure, prior_probs=None):
        super().__init__()
        self.samples = samples
        self.sample_dim = samples.shape[-1]
        self.labels = labels
        self.C = labels.shape[-1]
        self.conditional_model = FlowConditionalDensityEstimation(samples, labels, structure)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.C) / self.C)
        else:
            self.prior_log_probs = torch.log(prior_probs)

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def loss(self, samples, labels):
        return -torch.sum(self.log_prob(samples)* labels)

    def train(self, epochs, batch_size=None, unlabeled_samples=None, unlabeled_labels=None,
              test_samples=None, test_labels=None, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
        self.conditional_model.initialize_with_EM(torch.cat([self.samples, unlabeled_samples], dim=0), 50)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        samples = torch.cat([self.samples, unlabeled_samples])
        labels = torch.cat([self.labels, torch.ones(unlabeled_samples.shape[0], self.C) / self.C])
        batch_size = samples.shape[0]
        dataset = torch.utils.data.TensorDataset(samples, labels)
        train_accuracy_trace = []
        unlabeled_accuracy_trace = []
        test_accuracy_trace = []
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if False:
            #if __ % recording_frequency == 0 or __<100:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0], batch[1]) for _, batch in enumerate(dataloader)]).sum().item()
                    train_accuracy = compute_accuracy(self.log_prob(self.samples), self.labels)
                    train_accuracy_trace.append(train_accuracy.item())
                    unlabeled_accuracy = compute_accuracy(self.log_prob(unlabeled_samples), unlabeled_labels)
                    unlabeled_accuracy_trace.append(unlabeled_accuracy.item())
                    test_accuracy = compute_accuracy(self.log_prob(test_samples), test_labels)
                    test_accuracy_trace.append(test_accuracy.item())
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 4)) + '; device = ' + str(
                        device) + '; train_acc = ' + str(train_accuracy) + '; unlab_acc = ' + str(
                        unlabeled_accuracy) + '; test_acc= ' + str(test_accuracy))
        return train_accuracy_trace, unlabeled_accuracy_trace, test_accuracy_trace


structure = [[ConditionalRealNVPLayer,{'hidden_dims': [64,64,64], 'K':2}] for i in range(5)]
classifier = GenerativeClassifierSemiSupervised(samples, labels,structure)
train_accuracy_trace, unlabeled_accuracy_trace, test_accuracy_trace = classifier.train(5000,6000,unlabeled_samples, unlabeled_labels, test_samples, test_labels, recording_frequency = 10)
