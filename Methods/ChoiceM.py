import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel


def covariance(Y_tr):
    return np.cov(Y_tr.T)


def L_quantile(p):
    A = np.diag(np.ones(p - 1), k=-1) + np.diag(np.ones(p - 1), k=1)
    D = 2 * np.eye(p)
    D[0, 0] = 1
    D[-1, -1] = 1
    return D - A


def M_quantile(probs, gamma=None):
    return rbf_kernel(probs.reshape((-1, 1)), gamma=gamma)


def M_rbf(Y, gamma=None):
    return rbf_kernel(Y.T, gamma=gamma)


def L_all_linked(p):
    D = p * np.eye(p)
    A = np.ones((p, p))
    return D - A


def M_mu(p, L, mu):
    return np.linalg.inv(mu * L + (1 - mu) * np.eye(p))