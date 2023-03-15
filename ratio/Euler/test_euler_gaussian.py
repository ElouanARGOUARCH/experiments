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
num_samples = 30000
cat = torch.distributions.Categorical(probs = vector_density)
categorical_samples = cat.sample([num_samples])
target_samples = torch.cat([((categorical_samples%columns + torch.rand(num_samples))/columns).unsqueeze(-1),((1-(categorical_samples//columns + torch.rand(num_samples))/lines)).unsqueeze(-1)], dim = -1)

mean = torch.mean(target_samples, dim = 0)
cov = torch.cov(target_samples.T)
proposed_samples = torch.distributions.MultivariateNormal(mean, (cov + cov.T)/2).sample([num_samples])

from classifiers import *
samples = torch.cat([target_samples, proposed_samples], dim =0)
labels = torch.cat([torch.ones(num_samples), torch.zeros(num_samples)], dim = 0).long()
binary_classif = Classifier(2,samples, labels, [128,128,128])
binary_classif.train(500,3000,lr = 5e-3,weight_decay  = 0 , verbose =True)