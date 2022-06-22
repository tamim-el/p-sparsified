import numpy as np
from Utils.load_data import load_rf2
from Methods.VectorialModel import VectorialModel
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
X_tr, Y_tr, X_te, Y_te = load_rf2()
n_tr = X_tr.shape[0]
n_te = X_te.shape[0]
d = Y_tr.shape[1]

######## Non-sketched model #######################################################################

print('Non-sketched model in process...')

# Hyperparameters priorly obtained by inner 5-folds cv
L_wo_S = 2.06913808111479e-06
gammas_wo_S = 7.847599703514607e-06
                                    
kernel = Gaussian_kernel(gammas_wo_S)

clf = VectorialModel(kernel=kernel, L=L_wo_S, algo='krr')

t0 = time()

# KRR ################################################################################
clf.fit(X_tr, Y_tr)

# Training time
time_wo_S = time() - t0

rrmse_test_wo_S = clf.rrmse(X_te, Y_te)
arrmse_test_wo_S = np.mean(rrmse_test_wo_S)

print('Results obtained without Sketching on rf2 dataset: ')
print('Average Relative Root Mean Squared Error: ' + str(arrmse_test_wo_S))
print('RRMSE for each target: ')
print('chsi2: ' + str(rrmse_test_wo_S[0, 0]))
print('nasi2: ' + str(rrmse_test_wo_S[0, 1]))
print('eadm7: ' + str(rrmse_test_wo_S[0, 2]))
print('sclm7: ' + str(rrmse_test_wo_S[0, 3]))
print('clkm7: ' + str(rrmse_test_wo_S[0, 4]))
print('vali2: ' + str(rrmse_test_wo_S[0, 5]))
print('napm7: ' + str(rrmse_test_wo_S[0, 6]))
print('dldi4: ' + str(rrmse_test_wo_S[0, 7]))
print('Training lasted ' + str(time_wo_S) + ' s.')
print('\n')

######## sketched models ###########################################################################

# Hyperparameters priorly obtained by inner 5-folds cv
Ls = [0.0016237767391887243, 0.0016237767391887243, 5.455594781168515e-07]
gammas = [2.976351441631319e-07, 2.976351441631319e-07, 2.6366508987303555e-06]

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

    rrmse_test_S = np.zeros((n_rep, d))
    times_S = np.zeros(n_rep)
                                        
    for j in range(n_rep):

        S = choice_sketch(i, s, n_tr)

        kernel = Gaussian_kernel(gammas[i])

        clf = VectorialModel(kernel=kernel, Sketch=S, L=Ls[i], algo='krr')

        t0 = time()

        # KRR ################################################################################
        clf.fit(X_tr, Y_tr)

        # Training time
        times_S[j] = time() - t0

        rrmse_test_S[j, :] = clf.rrmse(X_te, Y_te)

    arrmse_mean = np.mean(np.mean(rrmse_test_S, axis=1), axis=0)
    arrmse_std = 0.5 * np.std(np.mean(rrmse_test_S, axis=1), axis=0)

    rrmse_test_mean = np.mean(rrmse_test_S, axis=0)
    rrmse_test_std = 0.5 * np.std(rrmse_test_S, axis=0)

    times_mean = np.mean(times_S)
    times_std = 0.5 * np.std(times_S)

    if i == 0:
        print('Results obtained with 20/n-SR sketch on rf2 dataset: ')
    elif i == 1:
        print('Results obtained with 20/n-SG sketch on rf2 dataset: ')
    else:
        print('Results obtained with Accumulation (m=20) on rf2 dataset: ')
    print('Average Relative Root Mean Squared Error: ' + str(arrmse_mean) + ' +- ' + str(arrmse_std))
    print('RRMSE for each target: ')
    print('chsi2: ' + str(rrmse_test_mean[0]) + ' +- ' + str(rrmse_test_std[0]))
    print('nasi2: ' + str(rrmse_test_mean[1]) + ' +- ' + str(rrmse_test_std[1]))
    print('eadm7: ' + str(rrmse_test_mean[2]) + ' +- ' + str(rrmse_test_std[2]))
    print('sclm7: ' + str(rrmse_test_mean[3]) + ' +- ' + str(rrmse_test_std[3]))
    print('clkm7: ' + str(rrmse_test_mean[4]) + ' +- ' + str(rrmse_test_std[4]))
    print('vali2: ' + str(rrmse_test_mean[5]) + ' +- ' + str(rrmse_test_std[5]))
    print('napm7: ' + str(rrmse_test_mean[6]) + ' +- ' + str(rrmse_test_std[6]))
    print('dldi4: ' + str(rrmse_test_mean[7]) + ' +- ' + str(rrmse_test_std[7]))
    print('Training lasted ' + str(times_mean) + ' +- ' + str(times_std) + ' s.')
    print('\n')