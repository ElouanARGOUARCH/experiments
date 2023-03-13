import torch
from matplotlib import image
import numpy


rgb = image.imread("euler.jpg")
lines, columns = rgb.shape[:-1]

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

#Sample data according to image
vector_density = grey.flatten()
vector_density = vector_density/torch.sum(vector_density)
lines, columns = grey.shape
num_samples = 300000
cat = torch.distributions.Categorical(probs = vector_density)
categorical_samples = cat.sample([num_samples])
target_samples = torch.cat([((categorical_samples%columns + torch.rand(num_samples))/columns).unsqueeze(-1),((1-(categorical_samples//columns + torch.rand(num_samples))/lines)).unsqueeze(-1)], dim = -1)

from density_estimation import *
linspace = 7
EM = FullRankGaussianMixtEM(target_samples,linspace**2)
EM.m = torch.cartesian_prod(torch.linspace(0,1, linspace),torch.linspace(0,1, linspace))
EM.train(200, verbose = True)
proposed_samples, labels = EM.sample([num_samples], joint = True)

from classifiers import *

labels = torch.cat([EM.K*torch.ones(num_samples).int(),labels])
samples = torch.cat([target_samples, proposed_samples], dim=0)
multi_classif = KClassifier(EM.K+1,samples, labels,[256,256,256])
for i in range(10):
    multi_classif.train(500,30000, lr=5e-4, weight_decay=5e-6, verbose = True)
    density_ratio = lambda samples : torch.exp(multi_classif.log_prob(samples))[:,-1]/torch.sum(torch.exp(multi_classif.log_prob(samples))[:,:-1], dim = -1)
    plot_2d_function(lambda x: density_ratio(x)*torch.exp(EM.log_prob(x)),range = [[0,1],[0,1]], bins = [lines, columns])
    plt.show()
