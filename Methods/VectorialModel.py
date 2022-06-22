import numpy as np
from scipy import linalg
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

class VectorialModel:

    def __init__(self, kernel, M=None, Sketch=None, L=1.,
                 algo='krr', alg_param=0.5, optim='adam', max_iter=1000,
                 lr=1e-3, tol=1e-3, monitoring_step=1,
                 early_stopping=False, j_early_stopping=None,
                 n_iter_no_change=5, verbose=False):
        # Saving parameters
        self.kernel = kernel
        self.M = M
        self.Sketch = Sketch
        self.L = L
        self.algo = algo
        self.alg_param = alg_param
        self.loss = choice_loss(algo=self.algo, param=self.alg_param)
        self.loss_vect = choice_loss_vect(algo=self.algo, param=self.alg_param)
        self.grad = choice_grad(algo=self.algo, param=self.alg_param)
        self.score = mrrmse
        self.optim = optim
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.monitoring_step = monitoring_step
        self.early_stopping = early_stopping
        self.j_early_stopping = j_early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose

    def fit(self, X_tr, Y_tr, X_val=None, Y_val=None):
        # Training
        self.X_tr = X_tr.copy()
        self.Y_tr = Y_tr.copy()
        self.n_tr = self.X_tr.shape[0]
        self.p = self.Y_tr.shape[1]

        if self.M is None:
            self.M = np.eye(self.p)

        if self.algo == 'krr':
            
            # Without sketching
            if self.Sketch is None:
                K_x = self.kernel(self.X_tr, Y=self.X_tr)
                B = K_x + self.n_tr * self.L * np.eye(self.n_tr)
                self.Omega = np.linalg.inv(B).dot(self.Y_tr)
            # With sketching
            else:
                S = self.Sketch
                K_x = S.multiply_Gram_both_sides(self.X_tr, self.kernel)
                K_x_S = S.multiply_Gram_one_side(self.X_tr, self.kernel, self.X_tr)
                B = K_x_S.T.dot(K_x_S) + self.n_tr * self.L * K_x
                B_inv = np.linalg.pinv(B)
                y_s = K_x_S.T.dot(self.Y_tr)
                self.Omega = B_inv.dot(y_s)

        elif self.algo in ['e_krr', 'e_svr', 'k_huber', 'mmr']:

            # Without sketching
            if self.Sketch is None:

                self.Omega, self.objectives, self.train_loss, self.val_loss, self.train_score, self.val_score = sgd(self.X_tr, self.Y_tr,
                                                                                                                X_val, Y_val, self.M,
                                                                                                                self.kernel, self.L, self.algo,
                                                                                                                self.loss_vect, self.grad, self.score,
                                                                                                                self.optim, self.lr,
                                                                                                                self.max_iter, self.tol,
                                                                                                                self.monitoring_step,
                                                                                                                self.early_stopping,
                                                                                                                self.j_early_stopping,
                                                                                                                self.n_iter_no_change,
                                                                                                                self.verbose)

            # With sketching
            else:

                S = self.Sketch
                
                self.Omega, self.objectives, self.train_loss, self.val_loss, self.train_score, self.val_score = sgd_sketch(self.X_tr, self.Y_tr,
                                                                                                                        X_val, Y_val,
                                                                                                                        self.kernel,
                                                                                                                        self.M, S,
                                                                                                                        self.L, self.algo,
                                                                                                                        self.loss_vect, self.grad,
                                                                                                                        self.score,
                                                                                                                        self.optim, self.lr,
                                                                                                                        self.max_iter, self.tol,
                                                                                                                        self.monitoring_step,
                                                                                                                        self.early_stopping,
                                                                                                                        self.j_early_stopping,
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


    def rrmse(self, X_te, Y_te):
        Y_pred = self.estimate_output_embedding(X_te)
        Y_tr_mean = np.mean(self.Y_tr, axis=0)
        denom = np.mean(np.power(Y_tr_mean - Y_te, 2), axis=0)
        num = mean_squared_error(Y_pred, Y_te, multioutput='raw_values').reshape((1, -1))
        return np.sqrt(num / denom)



def Huber(x, kappa):
    xnorm = np.linalg.norm(x)
    if xnorm <= kappa:
        return 0.5 * xnorm ** 2
    else:
        return kappa * (xnorm - 0.5 * kappa)

def Huber_vect(x, kappa):
    xnorm = np.linalg.norm(x, axis=1)
    res = 0.5 * xnorm ** 2
    res[np.where(xnorm > kappa)] = kappa * (xnorm[np.where(xnorm > kappa)] - 0.5 * kappa)
    return res

def grad_Huber(x, kappa):
    xnorm = np.linalg.norm(x)
    if xnorm <= kappa:
        return x
    else:
        return (kappa / xnorm) * x


def eL2(x, eps):
    xnorm = np.linalg.norm(x)
    if xnorm >= eps:
        return (xnorm - eps) ** 2
    else:
        return 0

def eL2_vect(x, eps):
    xnorm = np.linalg.norm(x, axis=1)
    res = np.zeros_like(x)
    res[np.where(xnorm >= eps)] = (xnorm[np.where(xnorm >= eps)] - eps) ** 2
    return res

def grad_eL2(x, eps):
    xnorm = np.linalg.norm(x)
    if xnorm >= eps:
        return 2 * ((xnorm - eps) / xnorm) * x
    else:
        return np.zeros_like(x)


def eL1(x, eps):
    xnorm = np.linalg.norm(x)
    if xnorm >= eps:
        return xnorm - eps
    else:
        return 0

def eL1_vect(x, eps):
    xnorm = np.linalg.norm(x, axis=1)
    res = np.zeros_like(xnorm)
    res[np.where(xnorm >= eps)] = xnorm[np.where(xnorm >= eps)] - eps
    return res

def grad_eL1(x, eps):
    xnorm = np.linalg.norm(x)
    if xnorm >= eps:
        return x / xnorm
    else:
        return np.zeros_like(x)


def Hinge(x):
    if x <= 1:
        return 1 - x
    else:
        return 0

def Hinge_vect(x):
    res = 1 - x
    res[np.where(x > 1)] = 0
    return res

def grad_Hinge(x):
    if x <= 1:
        return -1
    else:
        return 0


def mrrmse(Y_tr, Y_te, Y_pred, j=None):
    Y_tr_mean = np.mean(Y_tr, axis=0)
    denom = np.mean(np.power(Y_tr_mean - Y_te, 2), axis=0)
    num = mean_squared_error(Y_pred, Y_te, multioutput='raw_values').reshape((1, -1))
    if j is None:
        return np.mean(np.sqrt(num / denom))
    else:
        return np.sqrt(num / denom)[0, j]


def choice_loss(algo='mmr', param=None):
    if algo == 'mmr':
        def loss(x):
            return Hinge(x)
    elif algo == 'e_krr':
        def loss(x):
            return eL2(x, param)
    elif algo == 'e_svr':
        def loss(x):
            return eL1(x, param)
    else:
        def loss(x):
            return Huber(x, param)
    return loss


def choice_loss_vect(algo='mmr', param=None):
    if algo == 'mmr':
        def loss_vect(x):
            return Hinge_vect(x)
    elif algo == 'e_krr':
        def loss_vect(x):
            return eL2_vect(x, param)
    elif algo == 'e_svr':
        def loss_vect(x):
            return eL1_vect(x, param)
    else:
        def loss_vect(x):
            return Huber_vect(x, param)
    return loss_vect


def choice_grad(algo='mmr', param=None):
    if algo == 'mmr':
        def grad(x):
            return grad_Hinge(x)
    elif algo == 'e_krr':
        def grad(x):
            return grad_eL2(x, param)
    elif algo == 'e_svr':
        def grad(x):
            return grad_eL1(x, param)
    else:
        def grad(x):
            return grad_Huber(x, param)
    return grad


def sgd_sketch(X_tr, Y_tr, X_val, Y_val, kernel, M, S,
               L, algo, loss_vect, grad, score, optim, lr, max_iter,
               tol=1e-3, monitoring_step=100, early_stopping=False, j_early_stopping=None,
               n_iter_no_change=5,
               verbose=False):
    """
        Stochastic Gradient Descent for linear primal pb with sketched feature maps
    """
    # Computation of sketched feature maps
    p = Y_tr.shape[1]
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
            if algo == 'mmr':
                grad_l = grad(yi.dot(kS.dot(Omega).dot(M)))
                gradient = L * SKST.dot(Omega).dot(M) + grad_l * kS.reshape((-1, 1)).dot(yi.reshape((1, -1))).dot(M)
            else:
                grad_l = grad(kS.dot(Omega).dot(M) - yi)
                gradient = L * SKST.dot(Omega).dot(M) + kS.reshape((-1, 1)).dot(grad_l.reshape((1, -1))).dot(M)
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
                if algo == 'mmr':
                    predy_tr = pred_tr.dot(Y_tr.T)
                else:
                    predy_tr = pred_tr - Y_tr
                losses = loss_vect(predy_tr)
                train_loss.append(np.mean(losses))
                objectives.append(train_loss[-1] + (L / 2) * np.trace(SKST.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = KvalS.dot(Omega).dot(M)
                    if algo == 'mmr':
                        predy_val = pred_val.dot(Y_val.T)
                    else:
                        predy_val = pred_val - Y_val
                    losses = loss_vect(predy_val)
                    val_loss.append(np.mean(losses))

                # Computation of train and validation score
                train_score.append(score(Y_tr, Y_tr, pred_tr, j_early_stopping))
                if early_stopping:
                    val_score.append(score(Y_tr, Y_val, pred_val, j_early_stopping))                    

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


def sgd(X_tr, Y_tr, X_val, Y_val, M, kernel, L,
        algo, loss_vect, grad, score, optim, lr, max_iter,
        tol=1e-3, monitoring_step=100, early_stopping=False, j_early_stopping=None,
        n_iter_no_change=5,
        verbose=False):
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
    p = Y_tr.shape[1]
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
            if algo == 'mmr':
                grad_l = grad(yi.dot(k.dot(Omega).dot(M).T))
                gradient = L * K.dot(Omega).dot(M) + grad_l * k.reshape((-1, 1)).dot(yi.reshape((1, -1))).dot(M)
            else:
                grad_l = grad(k.dot(Omega).dot(M) - yi)
                gradient = L * K.dot(Omega).dot(M) + k.reshape((-1, 1)).dot(grad_l.reshape((1, -1))).dot(M)
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
                if algo == 'mmr':
                    predy_tr = pred_tr.dot(Y_tr.T)
                else:
                    predy_tr = pred_tr - Y_tr
                losses = loss_vect(predy_tr)
                train_loss.append(np.mean(losses))
                objectives.append(train_loss[-1] + (L / 2) * np.trace(K.dot(Omega).dot(M).dot(Omega.T)))

                # Computation of validation loss
                if early_stopping:
                    pred_val = K_valtr.dot(Omega).dot(M)
                    if algo == 'mmr':
                        predy_val = pred_val.dot(Y_val.T)
                    else:
                        predy_val = pred_val - Y_val
                    losses = loss_vect(predy_val)
                    val_loss.append(np.mean(losses))

                # Computation of train and validation score
                train_score.append(score(Y_tr, Y_tr, pred_tr, j_early_stopping))
                if early_stopping:
                    val_score.append(score(Y_tr, Y_val, pred_val, j_early_stopping))

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

                            if np.abs(val_score[1] - val_score[0]) == 0:
                                norm_es = 1
                            else:
                                norm_es = np.abs(val_score[1] - val_score[0])

                            if False not in (np.asarray(val_score[-n_iter_no_change - 1 : -1]) - np.asarray(val_score[-n_iter_no_change:]) < tol * norm_es):
                            #if False not in (np.asarray(val_score[-n_iter_no_change - 1 : -1]) - best_val_score < tol * norm_es):
                                if verbose:
                                    print("Early stopping attained at epoch: " + str(iter))
                                stop = True
                                break

        if stop:
            break

    return Omega, objectives, train_loss, val_loss, train_score, val_score