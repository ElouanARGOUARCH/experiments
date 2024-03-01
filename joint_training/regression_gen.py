import torch
import matplotlib.pyplot as plt
from markov_chain_monte_carlo import *
from IPython.display import clear_output
from utils import *
from conditional_density_estimation import *
from misc import *
torch.manual_seed(0)

sigma=.125
D_x = torch.cat([torch.linspace(-7.5,-2,1000),torch.linspace(-2,2,1000),torch.linspace(2,7.5,1000)])
D_y = 2*torch.cos(D_x) + sigma * torch.randn_like(D_x)
x0 = torch.rand(1)*15 - 7.5
y0 = 2*torch.cos(x0) + sigma * torch.randn_like(x0)

mask = torch.abs(D_x)>2
unlabeled_obs = D_y[~mask]
unlabeled_labels = D_x[~mask]
D_x = D_x[mask]
D_y = D_y[mask]

fig = plt.figure(figsize = (25,7))
ax = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]}, squeeze = True)
tt = torch.linspace(-7.5,7.5,1000)
ax[0].plot(tt, 2*torch.cos(tt), color = 'C0')
ax[0].fill_between(tt,  2*torch.cos(tt) - 3*sigma,  2*torch.cos(tt) + 3*sigma, color = 'C0', alpha = .25)
ax[0].scatter(D_x, D_y, color = 'C1', alpha = .5)
ax[0].scatter(unlabeled_labels, unlabeled_obs, color = 'C4', alpha = .5)
ax[0].axvline(x0.numpy(), linestyle = '--', color = 'green')
ax[0].scatter(x0,y0, color = 'green')

posterior = lambda x0: torch.exp(torch.distributions.Normal(2*torch.cos(x0), sigma).log_prob(y0))
ax[1].axvline(x0.numpy(), linestyle = '--', color = 'green')
plot_1d_unormalized_function(posterior, range = (-7.5, 7.5), bins = 500, color = 'grey', linestyle = '--')
plt.subplots_adjust(wspace=0.05, hspace=0)

total_models = []
D_y_plus = D_y.unsqueeze(-1)
D_x_plus = D_x.unsqueeze(-1)
prior_x0 = Uniform(torch.tensor([-7.5]), torch.tensor([7.5]))
prior_unlabeled = Uniform(torch.tensor([-2]), torch.tensor([2]))
for _ in range(10):
    for __ in range(5):
        generative_model = ConditionalDIF(D_y_plus,D_x_plus,3,[32,32,32])
        generative_model.initialize_with_EM(100)
        generative_model.train(100,500,lr = 5e-3, verbose = True)
        generative_model.train(100,500,lr = 5e-4, verbose = True)
        generative_model.train(100,500,lr = 5e-5, verbose = True)

        x0_log_posterior =lambda x, observations: generative_model.log_prob(observations,x)+ prior_x0.log_prob(x)
        x0_sampler = IndependentMetropolisHastingsMultipleObs(x0_log_posterior,y0.unsqueeze(-1),1, prior_x0)
        x0_samples = x0_sampler.sample(100, verbose = True)

        unlabeled_log_posterior =lambda x, observations: generative_model.log_prob(observations,x)+ prior_unlabeled.log_prob(x)
        unlabeled_sampler = IndependentMetropolisHastingsMultipleObs(unlabeled_log_posterior,unlabeled_obs.unsqueeze(-1),1, prior_unlabeled)
        unlabeled_labels = unlabeled_sampler.sample(100, verbose = True)
        D_x_plus = torch.cat([D_x.unsqueeze(-1), x0_samples, unlabeled_labels], dim=0)
        D_y_plus = torch.cat([D_y.unsqueeze(-1), y0.unsqueeze(-1), unlabeled_obs.unsqueeze(-1)], dim = 0)
    torch.save(generative_model, 'regression_models/regress_gen' + str(- + 1) + '.pt')

def plot_average_likelihood_function(log_likelihood_list, range = [[-10,10],[-10,10]], bins = [50,50], levels = 2 , alpha = 0.7, figsize = (10,6), show = True):
    total_log_likelihood = lambda y, x: torch.logsumexp(
        torch.cat([log_likelihood(y, x).unsqueeze(0) for log_likelihood in log_likelihood_list], dim=0), dim=0)
    plot_likelihood_function(total_log_likelihood, range=range, bins = bins, levels = levels, alpha = alpha, figsize = figsize, show = show)

plot_average_likelihood_function([model.log_prob for model in total_models], range = ((-10,10), (-2,2)), bins= (200,200), levels = 20)