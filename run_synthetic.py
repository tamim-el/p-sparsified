import sys
import numpy as np
import pandas as pd
from Utils.load_data import toy_dataset_1_perturbed
from Utils.create_df import create_df_plots
from Methods.ScalarModel import ScalarModel
from Methods.ScalarModelRFF import ScalarModelRFF
from Methods.Sketch import SubSample, Gaussian, pSparsified, Accumulation
from Methods.RFF import GaussianRFF
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from time import time

import seaborn as sns

def choice_sketch(i, s, n, type_pS):
    if i == 0:
        return SubSample((s, n))
    elif i == 1:
        p = 5.0 / n
        return pSparsified((s, n), p=p, type=type_pS)
    elif i == 2:
        p = 10.0 / n
        return pSparsified((s, n), p=p, type=type_pS)
    elif i == 3:
        p = 30.0 / n
        return pSparsified((s, n), p=p, type=type_pS)
    elif i == 4:
        p = 50.0 / n
        return pSparsified((s, n), p=p, type=type_pS)
    elif i == 5:
        return Gaussian((s, n))
    else:
        return Accumulation((s, n), m=30)


def choice_lr(i):
    if i == 1:
        lr = 2 * 1e-5
        return lr
    elif i == 2:
        lr = 5 * 1e-5
        return lr
    elif i == 3:
        lr = 1e-4
        return lr
    elif i == 4:
        lr = 5 * 1e-4
        return lr
    elif i == 7:
        lr = 3 * 1e-2
        return lr
    else:
        lr = 5 * 1e-2
        return lr


# Choice of loss
algo = sys.argv[1]


# Choice of p-Sparsified sketch type
type_pS = sys.argv[2]


# Setting random seed
np.random.seed(seed=42)


# Defining Gaussian kernel
def Gaussian_kernel(gamma):
    def Compute_Gram(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)
    return Compute_Gram


# Generating the training and test datasets
d = 10
n_tr = 10000
n_te = 10000
X_te, Y_true_te, Y_te = toy_dataset_1_perturbed(n_te, d)
X_tr, Y_true_tr, Y_tr = toy_dataset_1_perturbed(n_tr, d)

# Normalisation for MSE score
norm_mse = np.max(Y_te) - np.min(Y_te)

if type_pS == 'Rademacher':
    approx_types = ['Nystrom',
                'p-SR (p=5/n)', 'p-SR (p=10/n)', 'p-SR (p=30/n)', 'p-SR (p=50/n)',
                'Gaussian',
                'Accumulation (m=30)',
                'RFF']
else:
    approx_types = ['Nystrom',
                'p-SG (p=5/n)', 'p-SG (p=10/n)', 'p-SG (p=30/n)', 'p-SG (p=50/n)',
                'Gaussian',
                'Accumulation (m=30)',
                'RFF']

# List of dataframes containing numerical results
dfs = []

# For all types of approximation chosen
for i in range(8):

    approx_type = approx_types[i]
    print(approx_type + ' in process...')

    # Sketching sizes
    sketch_sizes = np.linspace(40, 140, num=11)
    n_sizes = len(sketch_sizes)

    # Learning parameters
    max_iter = 25
    lr = choice_lr(i)
    n_iter_no_change = 5
    monitoring_step = n_tr
    tol = None

    # Hyperparameters obtained by inner 5-folds cv
    # if Sketching
    if i < 7:
        # If loss is Huber
        if algo == 'k_huber':
            best_L_algo = 1e-07
            best_gamma_algo = 0.1
            best_param_algo = 100.0
        # If loss is SVR
        else:
            best_L_algo = 1e-08
            best_gamma_algo = 0.1
            best_param_algo = 4.6415888336127775
    # if RFF
    else:
        # If loss is Huber
        if algo == 'k_huber':
            best_L_algo = 1e-07
            best_gamma_algo = 1.0
            best_param_algo = 100.0
        # If loss is SVR
        else:
            best_L_algo = 1e-11
            best_gamma_algo = 1.0
            best_param_algo = 10.0

    # Number of replicates
    n_rep = 30

    mse_test_algo = np.zeros((n_sizes, n_rep))
    times_algo = np.zeros((n_sizes, n_rep))

    # if Sketching
    if i < 7:

        for i_s, ss in enumerate(sketch_sizes):
                                        
            for j in range(n_rep):

                s = int(ss)

                S = choice_sketch(i, s, n_tr, type_pS=type_pS)

                kernel = Gaussian_kernel(best_gamma_algo)

                clf = ScalarModel(kernel=kernel, Sketch=S, L=best_L_algo,
                                        algo=algo, alg_param=best_param_algo,
                                        lr=lr, tol=tol,
                                        max_iter=max_iter,
                                        monitoring_step=monitoring_step,
                                        n_iter_no_change=n_iter_no_change,
                                        early_stopping=False,
                                        score=mean_squared_error,
                                        verbose=False)

                t0 = time()

                # Kappa-Huber ################################################################################
                clf.fit(X_tr, Y_tr)

                # Training time
                times_algo[i_s, j] = time() - t0

                # Test mse
                Y_pred = clf.predict(X_te)
                mse_test_algo[i_s, j] = mean_squared_error(Y_te, Y_pred) / norm_mse


    # if RFF
    else:

        for i_s, ss in enumerate(sketch_sizes):
                                    
            for j in range(n_rep):

                s = int(ss / 2)
                
                RFF = GaussianRFF(d, s, best_gamma_algo)

                kernel = Gaussian_kernel(best_gamma_algo)

                clf = ScalarModelRFF(kernel=kernel, RFF=RFF, L=best_L_algo,
                                        algo=algo, alg_param=best_param_algo,
                                        lr=lr, tol=tol,
                                        max_iter=max_iter,
                                        monitoring_step=monitoring_step,
                                        n_iter_no_change=n_iter_no_change,
                                        early_stopping=False,
                                        score=mean_squared_error,
                                        verbose=False)

                t0 = time()

                # Kappa-Huber ################################################################################
                clf.fit(X_tr, Y_tr)

                # Training time
                times_algo[i_s, j] = time() - t0

                # Test mae
                Y_pred = clf.predict(X_te)
                mse_test_algo[i_s, j] = mean_squared_error(Y_te, Y_pred) / norm_mse


    df = create_df_plots(sketch_sizes=sketch_sizes,
                         n_rep=n_rep,
                         sketch_type=approx_type,
                         mse_test=mse_test_algo,
                         times=times_algo)

    dfs.append(df)

    print("Done!")


if algo == 'k_huber':
    algo_title_plot = 'k-huber'
else:
    algo_title_plot = 'e-svr'

if type_pS == 'Rademacher':
    pS_title_plot = 'p-SR'
    palette = 'PuRd'
    color = 'm'
else:
    pS_title_plot = 'p-SG'
    palette = 'Greys'
    color = 'k'

### First plot: Test MSE w.r.t. sketch size s #####################################################################################################

dfs_curve_1 = [dfs[0], dfs[5]]
df_curve_1 = pd.concat(dfs_curve_1)

dfs_curve_2 = dfs[1:5]
df_curve_2 = pd.concat(dfs_curve_2)

path_plot_mse = 'Plots/mse_' + algo_title_plot + '_' + pS_title_plot + '.pdf'

fig, ax = plt.subplots(figsize=(15, 12))
sns.lineplot(data=df_curve_1, x="Feature map size", y="Test MSE", ci='sd', hue="Sketch type", style="Sketch type", markers=["v", "^"], linewidth=5, markersize=20)
sns.lineplot(data=df_curve_2, x="Feature map size", y="Test MSE", ci='sd', hue="Sketch type", style="Sketch type", markers=True, linewidth=5, markersize=20, palette=palette)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_xticklabels([str(i)[:4] for i in ax.get_xticks()], fontsize = 25)
ax.set_yticklabels([str(i)[:4] for i in ax.get_yticks()], fontsize = 25)
plt.setp(ax.get_legend().get_texts(), fontsize='30')
plt.tight_layout()
plt.savefig(path_plot_mse, transparent=True)
plt.close()


### Second plot: Training times in s w.r.t. sketch size s #####################################################################################################

path_plot_time = 'Plots/time_' + algo_title_plot + '_' + pS_title_plot + '.pdf'

fig, ax = plt.subplots(figsize=(15, 12))
sns.lineplot(data=df_curve_1, x="Feature map size", y="Training times in s", ci='sd', hue="Sketch type", style="Sketch type", markers=["v", "^"], linewidth=5, markersize=20)
sns.lineplot(data=df_curve_2, x="Feature map size", y="Training times in s", ci='sd', hue="Sketch type", style="Sketch type", markers=True, linewidth=5, markersize=20, palette=palette)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_xticklabels([str(i)[:4] for i in ax.get_xticks()], fontsize = 25)
ax.set_yticklabels([str(i)[:4] for i in ax.get_yticks()], fontsize = 25)
plt.setp(ax.get_legend().get_texts(), fontsize='30')
plt.tight_layout()
plt.savefig(path_plot_time, transparent=True)
plt.close()


### Third plot: Test MSE w.r.t. training time in s #####################################################################################################

path_plot_compare = 'Plots/compare_' + algo_title_plot + '_' + pS_title_plot + '.pdf'

dfs_curve_3 = [dfs[0], dfs[5], dfs[-1]]
df_curve_3 = pd.concat(dfs_curve_3)

df_curve_4 = dfs[3]

df_curve_5 = dfs[-2]

fig, ax = plt.subplots(figsize=(20, 12))
sns.scatterplot(data=df_curve_3, x="Training times in s", y="Test MSE", ci=None, hue="Sketch type", style="Sketch type", markers=["v", "^", "D"], s=200)
sns.scatterplot(data=df_curve_4, x="Training times in s", y="Test MSE", ci=None, hue="Sketch type", style="Sketch type", markers='s', s=200, palette=[color])
sns.scatterplot(data=df_curve_5, x="Training times in s", y="Test MSE", ci=None, hue="Sketch type", style="Sketch type", markers='H', s=200, palette=['r'])
#plt.title("Training time")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_xticklabels([str(i)[:4] for i in ax.get_xticks()], fontsize = 30)
ax.set_yticklabels([str(i)[:4] for i in ax.get_yticks()], fontsize = 30)
plt.setp(ax.get_legend().get_texts(), fontsize='50')
plt.tight_layout()
plt.savefig(path_plot_compare, transparent=True)
plt.close()