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

mean = torch.mean(target_samples, dim = 0)
cov = torch.cov(target_samples.T)
proposed_samples = torch.distributions.MultivariateNormal(mean, (cov + cov.T)/2).sample([num_samples])

from classifiers import *
binary_classif = BinaryClassifier(target_samples, proposed_samples, [256,256,256])
binary_classif.train(500,30000,lr = 5e-4,weight_decay  = 5e-6, verbose =True)