import numpy as np
import matplotlib.pyplot as plt

def prior(mu):
    return np.ones(len(mu))

def verosimilitud(x,mu,sigma):
    norm = 1/(2*np.pi*sigma**2)**(1/2)
    return np.exp(-(x-mu)**2/(2*sigma**2))*norm

def mu_sigma(x_k,sigma_k):
    w_k = sigma_k**-2
    mu = np.sum(x_k*w_k)/np.sum(w_k)
    sigma = np.sum(w_k)**(-1/2)
    return mu,sigma
    
    
mu = np.linspace(-0,10,1000)
#x_k = np.linspace(-10,15,10)
#x_k = np.random.normal(loc=7,scale=2,size=20)
x_k = np.array([4.6, 6.0, 2.0, 5.8])
sigma_k = np.array( [2.0, 1.5, 5.0, 1.0])
#sigma_k = np.linspace(1E-3,2,len(x_k))


log_V = np.zeros(len(mu))

#print(verosimilitud(x_k[10],mu,sigma_k[10]))

for i in range(len(x_k)):
    log_V += np.log(verosimilitud(x_k[i],mu,sigma_k[i]))
    #print(V_i)

log_pos = log_V + np.log(prior(mu))
log_evidencia = np.amax(log_pos)

log_pos = log_pos-log_evidencia
posterior = np.exp(log_pos)
posterior = posterior/np.trapz(posterior,mu)

# Método 'manual' ---------------------------------------------
cero = np.argmax(log_pos)
mu_0 = mu[cero]
d2 = (log_pos[cero+1] - 2*log_pos[cero] + log_pos[cero-1]) / ((mu[cero]-mu[cero-1])**2)
sigma_0 = (-d2)**(-1/2)

# Método directo ----------------------------------------------
#mu_0,sigma_0 = mu_sigma(x_k,sigma_k)
print(mu_sigma(x_k,sigma_k))


plt.figure()
plt.plot(mu,posterior)
plt.xlabel(r'$\mu$'.format())
plt.ylabel('Probabilidad posterior')
plt.title(r'$\mu$ = {:.2f} $\pm$ {:.2f}'.format(mu_0,sigma_0))
plt.savefig('mean.png')