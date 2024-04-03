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
test_samples, test_labels = samples[:20000], labels[:20000].float()
train_samples, train_labels = samples[20000:60000], labels[20000:60000].float()

class GenerativeClassifier(torch.nn.Module):
    def __init__(self, sample_dim, C, structure):
        super().__init__()
        self.sample_dim = sample_dim
        self.C = C
        self.conditional_model = FlowConditionalDensityEstimation(samples, labels, structure)

    def compute_number_params(self):
        return self.conditional_model.compute_number_params()

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def log_posterior_prob(self, samples, prior):
        return torch.softmax(self.log_prob(samples) + torch.log(prior.unsqueeze(0)), dim=-1)

    def loss(self, samples, labels):
        return -torch.mean(self.conditional_model.log_prob(samples, labels))

    def train(self, epochs, batch_size, train_samples, train_labels,
              test_samples, test_labels, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
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
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    indices.append(__)
                    pbar.set_postfix_str('train_loss = ' + str(round(train_loss, 4)) + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices

sample_dim = train_samples.shape[-1]
C = train_labels.shape[-1]
structure= [[ConditionalRealNVPLayer,{'hidden_dims' : [64,64,64]}] for i in range(5)]
model_gen = GenerativeClassifier(sample_dim,C, structure)
print(model_gen.compute_number_params())
train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices = [],[],[],[]
for t in range(3):
    _ = model_gen.train(2000, int(train_samples.shape[0] / 20), train_samples, train_labels, test_samples, test_labels, lr=5e-4, recording_frequency=20)
    train_loss_trace+=_[0]
    unlabeled_loss_trace+=_[1]
    test_loss_trace+=_[2]
    indices+=_[3]
plt.plot(train_loss_trace)
plt.plot(unlabeled_loss_trace)
plt.plot(test_loss_trace)
plt.show()
print(compute_accuracy(model_gen.log_prob(train_samples), train_labels))
print(compute_accuracy(model_gen.log_prob(test_samples), test_labels))