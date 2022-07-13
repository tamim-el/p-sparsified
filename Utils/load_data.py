import numpy as np
import csv
import h5py
from skmultilearn.dataset import load_from_arff
import scipy.stats as st
from scipy.stats import norm
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import os
import glob


def toy_dataset_1_perturbed(n, d):
    """
        Load toy dataset from Friedman 4.3
    """
    X = np.random.uniform(size=(n, d))
    n_per = int(np.sqrt(n))
    indices = np.random.choice(n, n_per, replace=False)
    X_per = np.random.normal(loc=1.5, scale=0.25, size=(n_per, d))
    X[indices, :] = X_per.copy()

    Y_true = 0.1 * np.exp(4 * X[:, 0]) + 4 / (1 + np.exp(-20 * (X[:, 1] - 0.5))) + 3 * X[:, 2] + 2 * X[:, 3] + X[:, 4]

    Y_obs = Y_true + np.random.normal(size=n)

    return X, Y_true, Y_obs


def toy_quantile(n=50, t_min=0., t_max=1.5, noise=1., probs=[0.5]):
    """
    Parameters
    n: number of points (t, y)
    t_min: minimum input data t
    t_max: maximum input data t
    noise: noise level
    probs: probabilities (quantiles levels)

    Returns:
    x: sorted random data in [t_min, t_max]
    y: targets corresponding to x (following a noisy sin curve)
    q: true quantiles corresponding to x
    """
    t_down, t_up = 0., 1.5  # Bounds for the noise
    t = np.random.rand(n) * (t_max-t_min) + t_min
    t = np.sort(t)
    pattern = -np.sin(2*np.pi*t)  # Pattern of the signal
    enveloppe = 1 + np.sin(2*np.pi*t/3)  # Enveloppe of the signal
    pattern = pattern * enveloppe
    # Noise decreasing std (from noise+0.2 to 0.2)
    noise_std = 0.2 + noise*(t_up - t) / (t_up - t_down)
    # Gaussian noise with decreasing std
    add_noise = noise_std * np.random.randn(n)
    observations = pattern + add_noise
    quantiles = [pattern + norm.ppf(p, loc=np.zeros(n),
                                    scale=np.fabs(noise_std)) for p in probs]
    return t, observations, quantiles


def load_otoliths(filename_prefix_tr='Data/otoliths/train/Train_fc_',
                  training_info_path_tr='Data/otoliths/train/Training_info.csv',
                  filename_prefix_te='Data/otoliths/test/Test_fc_',
                  training_info_path_te='Data/otoliths/test/Testing_info.csv'):

    filename_suffix = '.npy'
    n = 3780
    X_train = np.zeros((n, 64 * 64))
    for i in range(n):
        X_train[i] = np.load(filename_prefix_tr + str(i) + filename_suffix)
    training_info = pd.read_csv(training_info_path_tr)
    Y_train = training_info['Ground truth'].values

    filename_suffix = '.npy'
    n = 165
    X_test = np.zeros((n, 64 * 64))
    for i in range(n):
        X_test[i] = np.load(filename_prefix_te + str(i) + filename_suffix)
    testing_info = pd.read_csv(training_info_path_te)
    Y_test = testing_info['Ground truth'].values

    return X_train, Y_train, X_test, Y_test


def load_rf1(path_tr='Data/rf1/rf1-train.arff', path_te='Data/rf1/rf1-test.arff', normalize=True):

    """
        Load Dataset rf1
    """
    x, y = load_from_arff(path_tr, label_count=8)
    X_tr, Y_tr = x.todense(), y.todense()

    x_test, y_test = load_from_arff(path_te, label_count=8)
    X_te, Y_te = x_test.todense(), y_test.todense()

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_rf2(path_tr='Data/rf2/rf2-train.arff', path_te='Data/rf2/rf2-test.arff', normalize=True):

    """
        Load Dataset rf2
    """

    x, y = load_from_arff(path_tr, label_count=8)
    X_tr, Y_tr = x.todense(), y.todense()

    x_test, y_test = load_from_arff(path_te, label_count=8)
    X_te, Y_te = x_test.todense(), y_test.todense()

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_scm1d(path_tr='Data/scm1d/scm1d-train.arff', path_te='Data/scm1d/scm1d-test.arff', normalize=True):

    """
        Load Dataset scm1d
    """

    x, y = load_from_arff(path_tr, label_count=16)
    X_tr, Y_tr = x.todense(), y.todense()

    x_test, y_test = load_from_arff(path_te, label_count=16)
    X_te, Y_te = x_test.todense(), y_test.todense()

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_scm20d(path_tr='Data/scm20d/scm20d-train.arff', path_te='Data/scm20d/scm20d-test.arff', normalize=True):

    """
        Load Dataset scm20d
    """

    x, y = load_from_arff(path_tr, label_count=16)
    X_tr, Y_tr = x.todense(), y.todense()

    x_test, y_test = load_from_arff(path_te, label_count=16)
    X_te, Y_te = x_test.todense(), y_test.todense()

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_sarcos(N=50, path_tr='Data/sarcos/sarcos_inv.mat', path_te='Data/sarcos/sarcos_inv_test.mat', normalize=True):

    """
        Load Dataset Sarcos
    """

    data_tr = loadmat(path_tr)
    x, y = data_tr['sarcos_inv'][:, :21].copy(), data_tr['sarcos_inv'][:, 21:].copy()
    idx = np.random.choice(x.shape[0], N * 7, replace=False)
    X_tr, Y_tr_full = x[idx].copy(), y[idx].copy()
    idx_t = np.empty(0)
    for i in range(7):
        idx_t = np.concatenate((idx_t, i * np.ones(N)))
    idx_t = np.random.permutation(idx_t)
    idx_t = idx_t.astype(int)
    Y_tr = np.choose(idx_t, Y_tr_full.T)
    lb = LabelBinarizer()
    lb.fit([1, 2, 3, 4, 5, 6, 7])
    T_tr = lb.transform(idx_t + 1)

    data_te = loadmat(path_te)
    X_te, Y_te = data_te['sarcos_inv_test'][:, :21].copy(), data_te['sarcos_inv_test'][:, 21:].copy()

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, T_tr, Y_tr, Y_tr_full, X_te, Y_te