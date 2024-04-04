import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from numpy.random import random

def creer_trajectoire(T,Q):
    x=np.zeros(T)
    x[0]=np.random.randn()

    for t in range(1,T):

                x[t]=0.5*x[t-1]+25*x[t-1]/(1+x[t-1]**2) + 8*np.cos(1.2*t) + np.sqrt(Q)*np.random.randn()
                #print(t)
    return x


def creer_observation(x,R):

    y=np.zeros(T)
    for t in range(0,T):
        y[t]=x[t]**2/20 + np.sqrt(R)*np.random.randn()

    return y

def filtrage_particulaire(particules,poids,y,t,Q,R,N):

    particules=0.5*particules + 25*particules/(1+particules**2) + 8*np.cos(1.2*t) + np.sqrt(Q)*np.random.randn(1,N)
   # print(particules)
    poids=norm.pdf(y,particules**2/20,np.sqrt(R))
  #  print(poids)
    poids=poids/np.sum(poids)
    estimateur=np.dot(particules,poids.T)
    
    index=multinomial_resample(poids)
    poids=np.ones(N)/N
    particules=particules[0,index]
        

    return (particules,poids,estimateur)


def multinomial_resample(weights):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.
   Parameters
   ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))



T=50
Q=10
R=1
N=5000

x=creer_trajectoire(T,Q)
y=creer_observation(x,R)

particules=np.random.randn(1,N)
#print(len(particules.T))
poids= np.ones(N)/N
estimateur=np.zeros(T)

for t in range(0,T):
##    result=filtrage_particulaire(particules,poids,y[t],t,Q,R,N)
##    particules=result[0]
##    poids=result[1]
##    estimateur[t]=result[2]
       [particules,poids,estimateur[t]]=filtrage_particulaire(particules,poids,y[t],t,Q,R,N)
    

plt.plot(range(T),x)
plt.plot(range(T),y)
plt.plot(range(T),estimateur,'green')
plt.show()

