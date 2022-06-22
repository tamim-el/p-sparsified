import numpy as np
from scipy import linalg

class QuantileModel:

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

    def fit(self, X_tr, Y_tr, Probs, X_val=None, Y_val=None):
        # Training
        self.X_tr = X_tr.copy()
        self.Y_tr = Y_tr.copy()
        self.Probs = Probs.copy()
        self.n_tr = self.X_tr.shape[0]
        self.p = self.Probs.shape[0]

        # Without sketching
        if self.Sketch is None:

            if self.M is None:
                self.M = np.eye(self.p)

            self.Omega, self.objectives, self.train_loss, self.val_loss = sgd(X_tr, Y_tr, Probs,
                                                                              X_val, Y_val, self.M,
                                                                              self.kernel, self.L,
                                                                              self.optim, self.lr,
                                                                              self.max_iter, self.tol,
                                                                              self.monitoring_step,
                                                                              self.early_stopping,
                                                                              self.n_iter_no_change,
                                                                              self.verbose)

        # With sketching
        else:

            S = self.Sketch
            self.Omega, self.objectives, self.train_loss, self.val_loss = sgd_sketch(X_tr, Y_tr, Probs,
                                                                                        X_val, Y_val,
                                                                                        self.kernel,
                                                                                        self.M, S,
                                                                                        self.L,
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


def Pinball(r, tau):
    return np.maximum(tau * r, (tau - 1) * r)

def Pinball_vect(r, tau):
    return np.sum(np.maximum(tau * r, (tau - 1) * r), axis=1)

def grad_Pinball(r, tau):
    res = tau * np.ones_like(r)
    res[np.where(r < 0)] -= 1
    return res


def Pinball_loss(Q_pred, Y_te, tau):
    return np.mean(Pinball_vect(Y_te.reshape((-1, 1)) - Q_pred, tau))


def Crossing_loss(Q_pred):
    q1 = Q_pred[:, :-1].copy()
    q2 = Q_pred[:, 1:].copy()
    diff = q2 - q1
    maxi = np.maximum(0, diff)
    sum_p = np.sum(maxi, axis=1)
    return np.mean(sum_p)



def sgd_sketch(X_tr, Y_tr, Probs, X_val, Y_val,
               kernel, M, S, L, optim, lr, max_iter,
               tol=1e-3, monitoring_step=100, early_stopping=False,
               n_iter_no_change=5, verbose=False):
    """
        Stochastic Gradient Descent for linear primal pb with sketched feature maps
    """
    # Computation of sketched feature maps
    p = Probs.shape[0]
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
            grad_l = grad_Pinball(yi - kS.reshape((1, -1)).dot(Omega.dot(M)), Probs)
            gradient = L * SKST.dot(Omega).dot(M) - kS.reshape((-1, 1)).dot(grad_l.reshape((1, -1))).dot(M)
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
                predy_tr = Y_tr.reshape((-1, 1)) - pred_tr
                losses = Pinball_vect(predy_tr, Probs)
                train_loss.append(np.mean(losses))
                objectives.append(train_loss[-1] + (L / 2) * np.trace(SKST.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = KvalS.dot(Omega).dot(M)
                    predy_val = Y_val.reshape((-1, 1)) - pred_val
                    losses = Pinball_vect(predy_val, Probs)
                    val_loss.append(np.mean(losses))

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

                            if np.abs(val_loss[1] - val_loss[0]) == 0:
                                norm_es = 1
                            else:
                                norm_es = np.abs(val_loss[1] - val_loss[0])

                            if False not in (np.asarray(val_loss[-n_iter_no_change - 1 : -1]) - np.asarray(val_loss[-n_iter_no_change:]) < tol * norm_es):
                                if verbose:
                                    print("Early stopping attained at epoch: " + str(iter))
                                stop = True
                                break

        if stop:
            break

    return Omega, objectives, train_loss, val_loss


def sgd(X_tr, Y_tr, Probs, X_val, Y_val,
        M, kernel, L, optim, lr, max_iter,
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
    p = Probs.shape[0]
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
            grad_l = grad_Pinball(yi - k.reshape((1, -1)).dot(Omega.dot(M)), Probs)
            gradient = L * K.dot(Omega).dot(M) - k.reshape((-1, 1)).dot(grad_l.reshape((1, -1))).dot(M)
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
                predy_tr = Y_tr.reshape((-1, 1)) - pred_tr
                losses = Pinball_vect(predy_tr, Probs)
                train_loss.append(np.mean(losses))
                objectives.append(train_loss[-1] + (L / 2) * np.trace(K.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = K_valtr.dot(Omega).dot(M)
                    predy_val = Y_val.reshape((-1, 1)) - pred_val
                    losses = Pinball_vect(predy_val, Probs)
                    val_loss.append(np.mean(losses))

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

                            #best_val_score = np.min(val_score)

                            if np.abs(val_loss[1] - val_loss[0]) == 0:
                                norm_es = 1
                            else:
                                norm_es = np.abs(val_loss[1] - val_loss[0])

                            if False not in (np.asarray(val_loss[-n_iter_no_change - 1 : -1]) - np.asarray(val_loss[-n_iter_no_change:]) < tol * norm_es):
                            #if False not in (np.asarray(val_score[-n_iter_no_change - 1 : -1]) - best_val_score < tol * norm_es):
                                if verbose:
                                    print("Early stopping attained at epoch: " + str(iter))
                                stop = True
                                break

        if stop:
            break

    return Omega, objectives, train_loss, val_loss