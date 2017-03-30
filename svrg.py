#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import time

# debugging
import ipdb
#ipdb.set_trace()

class SVRG:
    """SVRG SVM classifier.

    This class implements the Stochastic Variance Reduction Gradient method.
    currently only 2 classes are supported. Will be extended to multiclass
    using one vs all strategy at a later time.

    Parameters
    ----------
    M : int, default: 500
        Number of stochastic updates.

    nu : float, default: 1e-4
        Learning rate.

    lam : float, default: 1e-4
        Penalty associated with the size of our solution w.

    calc_loss : bool, default: False
        Specifies if the loss function is to be computed.

    Attributes
    ----------
    M,nu,lam,calc_loss as listed above

    w : array, shape (n_features,)
        Optimal weights.

    fit_time : float
        Total elapsed time of fitting y to dataset X in fit method.

    training_loss : (optional) array, shape(max ite or tot # of gradients computed)
        Loss function evaluated every 10th iteration.
    """
    def __init__(self, M=500, nu=1e-4, lam=1e-4, calc_loss=False):
        self.M = M
        self.nu = nu
        self.lam = lam
        self.calc_loss = calc_loss

    def fit(self, X, y):
        """Fit the model according to given training data. y*X of shape
        (n_samples, n_features) is cached as yX. W bar gradients are cached as
        grad_bar.

        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            Training vector.

        y : np.array, shape (n_smaples,)
            Target vector relative to X.

        Returns
        -------
        self : obejct
            Returns self.
        """
        self.fit_time = time.time()
        w = w_bar = np.zeros(X.shape[1])

        if self.calc_loss:
            loss = []

        yX = X*y[:,np.newaxis]
        k_tot = 0
        for s in range(X.shape[0]):

            mu = self._grad(yX,w)
            k_tot += X.shape[0]
            w = w_bar

            grad_bar = self._grad(yX, w_bar, cache=True)

            for t in range(self.M):

                i = np.random.choice(X.shape[0])

                grad_i = self._grad(yX[i],w)
                w += -self.nu * (grad_i - grad_bar[i].T + mu)

                k_tot += 2

            w_bar = w

            if self.calc_loss:
                if k_tot%10 == 0:
                    loss.append(self._loss(X,y,w))

            if k_tot > 100*X.shape[0]:
                break

        if self.calc_loss:
            self.training_loss = loss

        self.w = w
        self.fit_time = time.time() - self.fit_time

    def _grad(self, yX , w, cache=False):
        """Evaluates the gradient at a single sample, at entire sample set or
        caches gradients of all samples.

        Parameters
        ----------
        yX : np.array, shape (n_samples, n_features)
            Cached data struction where each row is y[i]*X[i,:].

        w : np.array, shape (n_features,)
            Current optimal solution.

        cache : bool, default: False
            Whether to compute a cached gradient matrix for all samples where
            each row is a gradient evaluated at that sample.

        Returns
        -------
        gradient : np.array, shape (n_features,) or (n_samples, n_features)
            Returns Sample gradient(s) or cached gradient matrix.
        """
        if yX.ndim == 1:
            grad = -yX.T if yX.dot(w) < 1 else np.zeros(len(w))
            return self.lam*w + grad
        else:
            m,n = yX.shape
            yXw = yX.dot(w)
            if cache:
                grad = np.zeros(yX.shape)
                for i in range(m):
                    if yXw[i] < 1:
                        grad[i] = -yX[i]
                return self.lam*w + grad
            grad = [-yX[i] if yXw[i] < 1 else np.zeros(n) for i in range(m)]
            return self.lam*w + 1/m*sum(grad).T


    def _loss(self, X, y, w):
        w_penalty = 0.5 * self.lam * norm(w)**2
        loss = np.mean(np.maximum(np.zeros(len(y)),1 - y*X.dot(w)))
        return w_penalty + loss

    def score(self, X, y):
        y_pred_rv = X.dot(self.w)
        y_pred = np.array([-1 if i < 0 else 1 for i in y_pred_rv])
        return np.mean(y_pred == y)

def main():
    import matplotlib.pyplot as plt
    from time import localtime, strftime

    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    for M_v in [1,10,100,500,1000,2000]:
        svrg = SVRG(M=M_v, calc_loss=True)
        svrg.fit(X,y)
        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.plot(svrg.training_loss)
        plt.savefig('../plots/svrg/loss_vs_ite_{}.pdf'.format(
            strftime("%Y.%m.%d_%H.%M.%S", localtime()) + '__M_{}'.format(M_v),
            format='pdf'))
        plt.close('all')
        print('Loss vs Iteration saved to ../plots/svrg')
        print('Time: {} in {} iterations'.format(
            round(svrg.fit_time,2), len(svrg.training_loss)*10))
        print('Final objective function val w/ M={}: {}\n'.format(
            svrg.M, round(svrg.training_loss[-1], 5)))

if __name__ == "__main__":
    main()
