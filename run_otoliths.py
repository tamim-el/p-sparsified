import numpy as np
from Utils.load_data import load_otoliths
from Methods.QuantileModel import QuantileModel, Pinball_loss, Crossing_loss
from Methods.ChoiceM import M_quantile
from Methods.Sketch import pSparsified, Accumulation
from sklearn.metrics.pairwise import rbf_kernel
from time import time


def choice_sketch(i, s, n):
    if i == 0:
        p = 20.0 / n
        return pSparsified((s, n), p=p, type='Rademacher')
    elif i == 1:
        p = 20.0 / n
        return pSparsified((s, n), p=p, type='Gaussian')
    else:
        return Accumulation((s, n), m=20)


# Setting random seed
np.random.seed(seed=42)


# Defining Gaussian kernel
def Gaussian_kernel(gamma):
    def Compute_Gram(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)
    return Compute_Gram


# Loading dataset
x_train, y_train, X_te, Y_te = load_otoliths()
# Splitting 10% of training set as validation set for the learning
n_tr = int(0.9 * len(x_train))
X_tr, X_val = x_train[:n_tr], x_train[n_tr:]
Y_tr, Y_val = y_train[:n_tr], y_train[n_tr:]
n_te = X_te.shape[0]

# Quantile levels to predict
probs = np.linspace(0.1, 0.9, 5)
d = len(probs)

# Learning parameters
lr = 1e-4
max_iter = 50
n_iter_no_change = 5
monitoring_step = n_tr

######## Non-sketched model #######################################################################

print('Non-sketched model in process...')

# Hyperparameters priorly obtained by inner 5-folds cv
L_wo_S = 0.0001
gamma_wo_S = 0.1
gamma_M_wo_S = 3.1622776601683795

                                    
kernel = Gaussian_kernel(gamma_wo_S)

M = M_quantile(probs, gamma=gamma_M_wo_S)

clf = QuantileModel(kernel=kernel, M=M, L=L_wo_S,
            max_iter=max_iter, lr=lr, monitoring_step=monitoring_step,
            early_stopping=True, n_iter_no_change=n_iter_no_change,
            verbose=False)

t0 = time()

# Quantile prediction ################################################################################
clf.fit(X_tr, Y_tr, probs, X_val, Y_val)

# Training time
time_wo_S = time() - t0

Q_pred = clf.predict(X_te)

score_pinball_wo_sketch = Pinball_loss(Q_pred, Y_te, probs)
score_crossing_wo_sketch = Crossing_loss(Q_pred)

print('Results obtained without Sketching on otoliths dataset: ')
print('Test pinball loss: ' + str(score_pinball_wo_sketch))
print('Test crossing loss: ' + str(score_crossing_wo_sketch))
print('Training lasted ' + str(time_wo_S) + ' s.')
print('\n')

######## sketched models ###########################################################################

# Hyperparameters priorly obtained by inner 5-folds cv
Ls = [0.0001, 0.0001, 1e-06]
gammas = [0.1, 0.1, 0.1]
gamma_Ms = [10.0, 10.0, 10.0]

# For all types of approximation chosen
for i in range(3):

    if i == 0:
        print('p-SR sketched-model in process...')
    elif i == 1:
        print('p-SG sketched-model in process...')
    else:
        print('Accumulation sketched-model in process...')

    # Number of replicates
    n_rep = 30

    # Sketch size
    s = 200

    pinballs_S = np.zeros(n_rep)
    crossings_S = np.zeros(n_rep)
    times_S = np.zeros(n_rep)
                                        
    for j in range(n_rep):

        S = choice_sketch(i, s, n_tr)

        kernel = Gaussian_kernel(gammas[i])

        M = M_quantile(probs, gamma=gamma_Ms[i])

        clf = QuantileModel(kernel=kernel, M=M, Sketch=S, L=Ls[i],
            max_iter=max_iter, lr=lr, monitoring_step=monitoring_step,
            early_stopping=True, n_iter_no_change=n_iter_no_change,
            verbose=False)

        t0 = time()

        # Quantile prediction ################################################################################
        clf.fit(X_tr, Y_tr, probs, X_val, Y_val)

        # Training time
        times_S[j] = time() - t0

        pinballs_S[j] = Pinball_loss(Q_pred, Y_te, probs)
        crossings_S[j] = Crossing_loss(Q_pred)

    pinball_mean = np.mean(pinballs_S)
    pinball_std = 0.5 * np.std(pinballs_S)

    crossing_mean = np.mean(crossings_S)
    crossing_std = 0.5 * np.std(crossings_S)

    times_mean = np.mean(times_S)
    times_std = 0.5 * np.std(times_S)

    if i == 0:
        print('Results obtained with 20/n-SR sketch on otoliths dataset: ')
    elif i == 1:
        print('Results obtained with 20/n-SG sketch on otoliths dataset: ')
    else:
        print('Results obtained with Accumulation (m=20) on otoliths dataset: ')
    print('Test pinball loss: ' + str(pinball_mean) + ' +- ' + str(pinball_std))
    print('Test crossing loss: ' + str(crossing_mean) + ' +- ' + str(crossing_std))
    print('Training lasted ' + str(times_mean) + ' +- ' + str(times_std) + ' s.')
    print('\n')