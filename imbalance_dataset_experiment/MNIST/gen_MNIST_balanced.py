import torch
from pre_processing import *
from targets import *
from tqdm import tqdm
from conditional_density_estimation import *
from misc.metrics import *

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
        return -torch.mean(torch.sum(self.log_prob(samples)*labels, dim = -1), dim = 0)

    def log_posterior_prob(self, samples, prior):
        log_joint = self.log_prob(samples) + torch.log(prior.unsqueeze(0))
        return log_joint - torch.logsumexp(log_joint, dim = -1, keepdim=True)

    def train(self, epochs, batch_size, train_samples, train_prior_probs, train_labels, test_samples, test_prior_probs, test_labels, recording_frequency=1, lr=5e-3,
              weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        current_samples = torch.cat([train_samples, test_samples])
        current_labels = torch.cat(
            [train_labels,test_prior_probs.unsqueeze(0).repeat(test_samples.shape[0], 1)])
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
                                         + '; test_loss = ' + str(round(test_loss, 4)) + '; device = ' + str(device))
        self.to(torch.device('cpu'))
        return aggregate_loss_trace, train_loss_trace, test_loss_trace

number_runs = 5
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
    torch.save(datasets,"gen_MNIST_balanced/datasets_" + str(run) + ".pt")

    sample_dim = train_samples.shape[-1]
    C = train_labels.shape[-1]
    structure = [[ConditionalRealNVPLayer, {'hidden_dims': [80, 80, 80]}] for i in range(4)] + [[ConditionalDIFLayer, {'hidden_dims': [32, 32], 'K': 3}] for i in range(1)]
    model_gen = GenerativeClassifierSemiSupervised(sample_dim,C, structure)
    model_gen.conditional_model.initialize_with_EM(torch.cat([train_samples, test_samples], dim=0),50, verbose=False)
    aggregate_loss_trace,train_loss_trace, test_loss_trace, indices = [],[],[],[]
    _ = model_gen.train(200, int(train_samples.shape[0] / 20), train_samples,train_prior_probs, train_labels, test_samples,test_prior_probs,test_labels, lr=5e-3, recording_frequency=10)
    print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(test_samples,test_prior_probs), test_labels))
    aggregate_loss_trace += _[0]
    train_loss_trace += _[1]
    test_loss_trace += _[2]
    _ = model_gen.train(200, int(train_samples.shape[0] / 20), train_samples, train_prior_probs, train_labels,
                        test_samples, test_prior_probs, test_labels, lr=1e-3, recording_frequency=10)
    print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(test_samples,test_prior_probs), test_labels))
    aggregate_loss_trace += _[0]
    train_loss_trace += _[1]
    test_loss_trace += _[2]
    _ = model_gen.train(200, int(train_samples.shape[0] / 20), train_samples, train_prior_probs, train_labels,
                        test_samples, test_prior_probs, test_labels, lr=5e-4, recording_frequency=10)
    print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(test_samples,test_prior_probs), test_labels))
    aggregate_loss_trace += _[0]
    train_loss_trace += _[1]
    test_loss_trace += _[2]
    _ = model_gen.train(200, int(train_samples.shape[0] / 20), train_samples, train_prior_probs, train_labels,
                        test_samples, test_prior_probs, test_labels, lr=1e-4, recording_frequency=10)
    print(compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels))
    print(compute_accuracy(model_gen.log_posterior_prob(test_samples,test_prior_probs), test_labels))
    aggregate_loss_trace += _[0]
    train_loss_trace += _[1]
    test_loss_trace += _[2]
    torch.save(aggregate_loss_trace, "gen_MNIST_balanced/aggregate_loss_trace_" + str(run) + ".pt")
    torch.save(train_loss_trace, "gen_MNIST_balanced/train_loss_trace_" + str(run) + ".pt")
    torch.save(test_loss_trace, "gen_MNIST_balanced/test_loss_trace_" + str(run) + ".pt")
    torch.save(model_gen,"gen_MNIST_balanced/model_gen_" +str(run) + ".pt")