import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv('surv-BR.csv')

df = df.dropna(subset=['DSS_MONTHS'])
label = df['DSS_MONTHS'].values


mu = [np.quantile(label, 0.25), np.quantile(label, 0.75)]
pi = [0.5, 0.5]
var = [np.var(label), np.var(label)]

n_samples = len(label)


for iter in range(200):
    r = np.zeros((n_samples, 2))

    for c, g, p in zip(range(2), [norm(loc=mu[0], scale=np.sqrt(var[0])), norm(loc=mu[1], scale=np.sqrt(var[1]))], pi):
        r[:, c] = p * g.pdf(label)

    for i in range(n_samples):
        r[i] /= (np.sum(pi) * np.sum(r, axis=1)[i])

    m_c = np.sum(r, axis=0)

    pi = m_c / n_samples
    mu = np.sum(label.reshape(n_samples, 1) * r, axis=0) / m_c
    var_c = []

    for c in range(2):
        var_c.append((1 / m_c[c]) * np.dot(((r[:, c].reshape(n_samples, 1)) * (label.reshape(n_samples, 1) - mu[c])).T, (label.reshape(n_samples, 1) - mu[c])))

    var = var_c


std_dev = np.sqrt(var).flatten()


x = np.linspace(label.min(), label.max(), num=1000)
plt.plot(x, pi[0] * norm.pdf(x, mu[0], std_dev[0]), 'bo', markersize=1)
plt.plot(x, pi[1] * norm.pdf(x, mu[1], std_dev[1]), 'ro', markersize=1)

plt.xticks(np.arange(x.min(), x.max(), step=50), fontsize=12)
plt.yticks(fontsize=13)

plt.xlabel("DSS Months", fontsize=13)
plt.show()