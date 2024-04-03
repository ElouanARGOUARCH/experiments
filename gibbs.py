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
unlabeled_samples, unlabeled_labels = samples[:5000], labels[:5000].float()
samples, labels = samples[5000:15000], labels[5000:15000].float()

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
        log_joint = self.log_prob(samples) + torch.log(prior.unsqueeze(0))
        return log_joint - torch.logsumexp(log_joint, dim = -1, keepdim=True)

    def loss(self, samples, labels):
        return -torch.mean(self.log_prob.log_prob(samples, labels))

    def train(self, epochs, batch_size, train_samples, train_labels,test_samples = [], test_prior_prob = [],recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
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
                    unlabeled_loss = self.loss(unlabeled_samples, unlabeled_labels).item()
                    unlabeled_loss_trace.append(unlabeled_loss)
                    test_loss = self.loss(test_samples, test_labels).item()
                    test_loss_trace.append(test_loss)
                    indices.append(__)
                    pbar.set_postfix_str('train_loss = ' + str(round(train_loss, 4)) + '; unlabeled_loss = ' + str(round(unlabeled_loss, 4)) + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(
                        device))
        self.to(torch.device('cpu'))
        return train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices

def train_with_Gibbs(T, epochs, batch_size, train_samples,train_prior_probs, train_labels, unlabeled_samples, unlabeled_prior_probs, unlabeled_labels, test_samples, test_prior_probs, test_labels, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
    model_gen = GenerativeClassifier(sample_dim, C, structure)
    train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices = model_gen.train(epochs, batch_size,train_samples,train_labels, unlabeled_samples, unlabeled_labels, test_samples, test_labels, recording_frequency, lr, weight_decay)
    print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(unlabeled_samples, unlabeled_prior_probs), unlabeled_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(test_samples, test_prior_probs), test_labels))
    current_unlabeled_labels = torch.distributions.Categorical(torch.exp(model_gen.log_posterior_prob(unlabeled_samples, unlabeled_prior_probs))).sample()
    current_labels = torch.cat([train_labels,torch.nn.functional.one_hot(current_unlabeled_labels, num_classes = C)])
    current_samples = torch.cat([train_samples,unlabeled_samples])
    aggregate_loss_trace = train_loss_trace
    for t in range(T):
        model_gen = GenerativeClassifier(sample_dim, C, structure)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_gen.to(device)
        para_dict = []
        for model in model_gen.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        dataset = torch.utils.data.TensorDataset(current_samples, current_labels)
        pbar = tqdm(range(epochs))
        list_models = []
        for __ in pbar:
            model_gen.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model_gen.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0:
                with torch.no_grad():
                    model_gen.to(torch.device('cpu'))
                    aggregate_loss = model_gen.loss(current_samples,current_labels).item()
                    aggregate_loss_trace.append(aggregate_loss)
                    train_loss = model_gen.loss(train_samples,train_labels).item()
                    train_loss_trace.append(train_loss)
                    unlabeled_loss = model_gen.loss(unlabeled_samples,unlabeled_labels).item()
                    unlabeled_loss_trace.append(unlabeled_loss)
                    test_loss = model_gen.loss(test_samples,test_labels).item()
                    test_loss_trace.append(test_loss)
                    pbar.set_postfix_str('aggregate_loss = ' + str(round(aggregate_loss, 4)) + '; device = ' + str(
                        device) + '; train_loss = ' + str(train_loss) + '; unlab_loss = ' + str(
                        unlabeled_loss) + '; test_loss= ' + str(test_loss))
        model_gen.to(torch.device('cpu'))
        print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
        print(compute_accuracy(model_gen.log_posterior_prob(unlabeled_samples, unlabeled_prior_probs), unlabeled_labels))
        print(compute_accuracy(model_gen.log_posterior_prob(test_samples, test_prior_probs), test_labels))
        current_unlabeled_labels = torch.distributions.Categorical(torch.exp(model_gen.log_posterior_prob(unlabeled_samples, unlabeled_prior_probs))).sample()
        current_labels = torch.cat([train_labels,torch.nn.functional.one_hot(current_unlabeled_labels, num_classes = C)])
    return aggregate_loss_trace, train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices

sample_dim = train_samples.shape[-1]
C = train_labels.shape[-1]
structure= [[ConditionalRealNVPLayer,{'hidden_dims' : [56,56,56]}] for i in range(4)] + [[ConditionalDIFLayer,{'hidden_dims': [20,20], 'K':3}]]
model_gen = GenerativeClassifier(sample_dim,C, structure)
print(model_gen.compute_number_params())
aggregate_loss_trace, train_loss_trace, unlabeled_loss_trace, test_loss_trace, indices = train_with_Gibbs(2,1001,int(train_samples.shape[0]/5),train_samples,train_prior_probs,train_labels,unlabeled_samples,unlabeled_prior_probs,unlabeled_labels, test_samples,test_prior_probs,test_labels, recording_frequency=20)
plt.plot(aggregate_loss_trace)
plt.plot(train_loss_trace)
plt.plot(unlabeled_loss_trace)
plt.plot(test_loss_trace)
plt.show()
