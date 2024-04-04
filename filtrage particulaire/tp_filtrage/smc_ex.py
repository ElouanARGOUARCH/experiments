import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def observations(T, Q, R):
    x = [np.random.randn()]
    y = [x[-1]**2 / 20 + R * np.random.randn()]
    
    for t in range(1,T):
        x.append(0.5 * x[-1] + 25 * x[-1] / (1 + x[-1]**2) + 8 * np.cos(1.2*t) + Q * np.random.randn())
        y.append(x[-1]**2 / 20 + R * np.random.randn())

    return x,y

def reechantillonnage(xt_1, wt_1, t, Q):
    n_particules = xt_1.shape[0]
    A  = np.random.choice(range(0,n_particules), n_particules, p = wt_1.flatten())
    xt = 0.5 * xt_1[A] + 25 * xt_1[A] / (1 + xt_1[A]**2) + 8 * np.cos(1.2*t)\
                + Q * np.random.randn(n_particules,1)
    return xt
    
def filtre_particulaire(obs, n_particules, Q, R, x0):
    T = len(obs)
    xt = np.random.randn(n_particules, 1)
    # xt = x0 * np.ones((n_particules, 1))
    wt = norm.pdf(obs[0], xt**2 / 20, R)
    wt = wt / wt.sum()
    x  = np.array(xt)
    w  = np.array(wt)


    for t in range(1,T):
        xt = reechantillonnage(xt, wt, t, Q)
        wt = norm.pdf(obs[t], xt**2 / 20, R)
        wt = wt / wt.sum()
    
        x = np.hstack([x, xt])
        w = np.hstack([w, wt])
        
    return x,w 

Q, R = np.sqrt(2), 1
x,y = observations(50, Q, R)
xis, w = filtre_particulaire(y, 5000, Q, R, x0 = x[0])
hidden_states = (xis * w).sum(0)
print(hidden_states.shape)
plt.figure(figsize = (10,5))
plt.plot(hidden_states, label = 'estimated signal')
plt.plot(x, label = 'real signal')
plt.legend(fontsize = 10)
plt.show()
# fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (14,10))
# ax[0].plot(range(0,50), x, linewidth = 2.5, color = 'red')
# ax[1].plot(range(0,50), y, linewidth = 2.5, color = 'royalblue')

# ax[0].set_title('Etats cachés', fontsize = 20)
# ax[1].set_title('Observations', fontsize = 20)