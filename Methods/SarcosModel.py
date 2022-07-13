import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error

class SarcosModel:

    def __init__(self, kernel, M=None, Sketch=None, L=1.,
                 optim='adam', max_iter=1000,
                 lr=1e-3, tol=1e-3, monitoring_step=1,
                 early_stopping=False, n_iter_no_change=5,
                 verbose=False):
        # Saving parameters
        self.kernel = kernel
        self.M = M
        self.Sketch = Sketch
        self.L = L
        self.optim = optim
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.monitoring_step = monitoring_step
        self.early_stopping =early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose

    def fit(self, X_tr, Y_tr, T_tr, Y_tr_full, X_val=None, Y_val=None, T_val=None, Y_val_full=None):
        # Training
        self.X_tr = X_tr.copy()
        self.Y_tr = Y_tr.copy()
        self.T_tr = T_tr.copy()
        self.Y_tr_full = Y_tr_full.copy()
        self.n_tr = self.X_tr.shape[0]
        self.p = self.T_tr.shape[1]

        if self.M is None:
            self.M = np.eye(self.p)

        # Without sketching
        if self.Sketch is None:

            self.Omega, self.objectives, self.train_loss, self.val_loss, self.train_score, self.val_score = sgd(X_tr, Y_tr, T_tr, Y_tr_full,
                                                                                                                X_val, Y_val, T_val, Y_val_full,
                                                                                                                self.kernel, self.M, self.L,
                                                                                                                self.optim, self.lr,
                                                                                                                self.max_iter, self.tol,
                                                                                                                self.monitoring_step,
                                                                                                                self.early_stopping,
                                                                                                                self.n_iter_no_change,
                                                                                                                self.verbose)

        # With sketching
        else:

            S = self.Sketch
            self.Omega, self.objectives, self.train_loss, self.val_loss, self.train_score, self.val_score = sgd_sketch(X_tr, Y_tr, T_tr, Y_tr_full,
                                                                                                                       X_val, Y_val, T_val, Y_val_full,
                                                                                                                       self.kernel, self.M, S, self.L,
                                                                                                                       self.optim, self.lr,
                                                                                                                       self.max_iter, self.tol,
                                                                                                                       self.monitoring_step,
                                                                                                                       self.early_stopping,
                                                                                                                       self.n_iter_no_change,
                                                                                                                       self.verbose)


    def estimate_output_embedding(self, X_te):
        # Without sketching
        if self.Sketch is None:
            K_x_te_tr = self.kernel(X_te, self.X_tr)
            Y_pred = K_x_te_tr.dot(self.Omega).dot(self.M)
        # With sketching
        else:
            S = self.Sketch
            KS = S.multiply_Gram_one_side(X_te, self.kernel, Y=self.X_tr)
            Y_pred = KS.dot(self.Omega).dot(self.M)
        return Y_pred

    
    def predict(self, X_te):
        Y_pred = self.estimate_output_embedding(X_te)
        return Y_pred


def nMSE(Y_true, Y_pred):
    return np.mean(np.linalg.norm(Y_pred - Y_true, axis=1) ** 2) / np.mean(np.linalg.norm(np.zeros_like(Y_true) - Y_true, axis=1) ** 2)


def sgd_sketch(X_tr, Y_tr, T_tr, Y_tr_full,
               X_val, Y_val, T_val, Y_val_full,
               kernel, M, S, L, optim, lr, max_iter,
               tol=1e-3, monitoring_step=100, early_stopping=False,
               n_iter_no_change=5, verbose=False):
    """
        Stochastic Gradient Descent for linear primal pb with sketched feature maps
    """
    # Computation of sketched feature maps
    p = T_tr.shape[1]
    KS = S.multiply_Gram_one_side(X_tr, kernel)
    SKST = S.multiply_Gram_both_sides(X_tr, kernel)
    # For validation set if early_stopping
    if early_stopping:
        KvalS = S.multiply_Gram_one_side(X_val, kernel, Y=X_tr)

    # Init weights
    s = SKST.shape[0]
    Omega = np.zeros((s, p))
    n_tr = X_tr.shape[0]

    # Init moments and exp decays if adam used
    if optim == 'adam':
        beta1, beta2 = 0.9, 0.999
        m, v = 0, 0
        eps = 1e-8

    # Init monitoring
    objectives = []
    train_loss = []
    val_loss = []
    train_score = []
    val_score = []

    count_monitoring = 0
    stop = False

    # Iterations over epochs
    for iter in range(max_iter):
        
        # Iterations over training set
        for i in range(n_tr):
            t = n_tr * iter + i + 1
            # Computation of gradient
            yi = Y_tr[i]
            kS = KS[i, :]
            eti = T_tr[i]
            coef = kS.reshape((1, -1)).dot(Omega).dot(M).dot(eti.reshape((-1, 1))) - yi
            gradient = L * SKST.dot(Omega).dot(M) - coef * kS.reshape((-1, 1)).dot(eti.reshape((1, -1))).dot(M)
            # Update
            if optim == 'sgd':
                Omega = Omega - lr * gradient
            else:
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
                mhat = m / (1 - (beta1 ** t))
                vhat = v / (1 - (beta2 ** t))
                vhattemp = np.power(vhat, 1/2) + eps
                Omega = Omega - lr * np.divide(mhat, vhattemp)

            # Monitoring
            if (i + 1) % monitoring_step == 0:
                count_monitoring += 1
                # Computation of train loss and objective function
                pred_tr = KS.dot(Omega).dot(M)
                predy_tr = (Y_tr - np.diag(pred_tr.dot(T_tr.T))) ** 2
                losses = 0.5 * np.sum(predy_tr)
                train_loss.append(losses)
                objectives.append(train_loss[-1] + (L / 2) * np.trace(SKST.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = KvalS.dot(Omega).dot(M)
                    predy_val = (Y_val - np.diag(pred_val.dot(T_val.T))) ** 2
                    losses = 0.5 * np.sum(predy_val)
                    val_loss.append(losses)

                # Computation of train and validation score
                score_tr = nMSE(Y_tr_full, pred_tr)
                train_score.append(score_tr)
                if early_stopping:
                    score_val = nMSE(Y_val_full, pred_val)
                    val_score.append(score_val)

                # Stopping criterion and early stopping
                if tol is not None:

                    if count_monitoring > n_iter_no_change:

                        if np.abs(objectives[1] - objectives[0]) == 0:
                            norm_crit = 1
                        else:
                            norm_crit = np.abs(objectives[1] - objectives[0])

                        # Stopping criterion for objective
                        if False not in (np.asarray(objectives[-n_iter_no_change - 1 : -1]) - np.asarray(objectives[-n_iter_no_change:]) < tol * norm_crit):
                            if verbose:
                                print("Stopping criterion attained at epoch: " + str(iter))
                            stop = True
                            break

                        # Early stopping for validation score
                        if early_stopping:

                            if np.abs(val_score[1] - val_score[0]) == 0:
                                norm_es = 1
                            else:
                                norm_es = np.abs(val_score[1] - val_score[0])

                            if False not in (np.asarray(val_score[-n_iter_no_change - 1 : -1]) - np.asarray(val_score[-n_iter_no_change:]) < tol * norm_es):
                                if verbose:
                                    print("Early stopping attained at epoch: " + str(iter))
                                stop = True
                                break

        if stop:
            break

    return Omega, objectives, train_loss, val_loss, train_score, val_score


def sgd(X_tr, Y_tr, T_tr, Y_tr_full,
        X_val, Y_val, T_val, Y_val_full,
        kernel, M, L, optim, lr, max_iter,
        tol=1e-3, monitoring_step=100, early_stopping=False,
        n_iter_no_change=5, verbose=False):
    """
        Stochastic Gradient Descent for primal pb without sketching
    """
    # Computation of K
    K = kernel(X_tr, X_tr)
    # For validation set if early_stopping
    if early_stopping:
        K_valtr = kernel(X_val, X_tr)

    # Init weights
    n_tr = X_tr.shape[0]
    p = T_tr.shape[1]
    Omega = np.zeros((n_tr, p))

    # Init moments and exp decays if adam used
    if optim == 'adam':
        beta1, beta2 = 0.9, 0.999
        m, v = 0, 0
        eps = 1e-8

    # Init monitoring
    objectives = []
    train_loss = []
    val_loss = []
    train_score = []
    val_score = []

    count_monitoring = 0
    stop = False

    # Iterations over epochs
    for iter in range(max_iter):
        
        # Iterations over training set
        for i in range(n_tr):
            t = n_tr * iter + i + 1
            # Computation of gradient
            yi = Y_tr[i]
            k = K[i, :]
            eti = T_tr[i]
            coef = k.reshape((1, -1)).dot(Omega).dot(M).dot(eti.reshape((-1, 1))) - yi
            gradient = L * K.dot(Omega).dot(M) - coef * k.reshape((-1, 1)).dot(eti.reshape((1, -1))).dot(M)
            # Update
            if optim == 'sgd':
                Omega = Omega - lr * gradient
            else:
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
                mhat = m / (1 - (beta1 ** t))
                vhat = v / (1 - (beta2 ** t))
                vhattemp = np.power(vhat, 1/2) + eps
                Omega = Omega - lr * np.divide(mhat, vhattemp)

            # Monitoring
            if (i + 1) % monitoring_step == 0:
                count_monitoring += 1
                # Computation of train loss and objective function
                pred_tr = K.dot(Omega).dot(M)
                predy_tr = (Y_tr - np.diag(pred_tr.dot(T_tr.T))) ** 2
                losses = 0.5 * np.sum(predy_tr)
                train_loss.append(np.mean(losses))
                objectives.append(train_loss[-1] + (L / 2) * np.trace(K.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = K_valtr.dot(Omega).dot(M)
                    predy_val = (Y_val - np.diag(pred_val.dot(T_val.T))) ** 2
                    losses = 0.5 * np.sum(predy_val)
                    val_loss.append(np.mean(losses))

                # Computation of train and validation score
                score_tr = nMSE(Y_tr_full, pred_tr)
                train_score.append(score_tr)
                if early_stopping:
                    score_val = nMSE(Y_val_full, pred_val)
                    val_score.append(score_val)

                # Stopping criterion and early stopping
                if tol is not None:

                    if count_monitoring > n_iter_no_change:

                        #best_objective = np.min(objectives)

                        if np.abs(objectives[1] - objectives[0]) == 0:
                            norm_crit = 1
                        else:
                            norm_crit = np.abs(objectives[1] - objectives[0])

                        # Stopping criterion for objective
                        if False not in (np.asarray(objectives[-n_iter_no_change - 1 : -1]) - np.asarray(objectives[-n_iter_no_change:]) < tol * norm_crit):
                        #if False not in (np.asarray(objectives[-n_iter_no_change:]) - best_objective < tol * norm_crit):
                            if verbose:
                                print("Stopping criterion attained at epoch: " + str(iter))
                            stop = True
                            break

                        # Early stopping for validation score
                        if early_stopping:

                            if np.abs(val_score[1] - val_score[0]) == 0:
                                norm_es = 1
                            else:
                                norm_es = np.abs(val_score[1] - val_score[0])

                            if False not in (np.asarray(val_score[-n_iter_no_change - 1 : -1]) - np.asarray(val_score[-n_iter_no_change:]) < tol * norm_es):
                                if verbose:
                                    print("Early stopping attained at epoch: " + str(iter))
                                stop = True
                                break

        if stop:
            break

    return Omega, objectives, train_loss, val_loss, train_score, val_score