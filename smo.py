#!/usr/bin/env python3

import numpy as np
import time

# debugging
import ipdb
#ipdb.set_trace()

class SMO:
    """SMO SVM classifier.

    This class implements the SMO-type decomposition popularized in the LIBSVM
    package. Currently only 2 classes are supported. Will be extended to
    multiclass using one vs all strategy at a later time.

    Parameters
    ----------
    C : int, default: 5
        Box constraints on the solution.

    Tau : float, default: 1e-5
        Error tolerance.

    max_ite : int, default: 500
        Maximum iterations to search for a solution.

    calc_loss : bool, default: False
        Specifies if the loss function is to be computed.

    Attributes
    ----------
    C,Tau,max_ite,calc_loss as listed above

    alpha : array, shape (n_samples)
        Dual optimal variable.

    fit_time : float
        Total elapsed time of fitting y to dataset X in fit method.

    training_loss : (optional) array, shape(max ite or tot # of gradients computed)
        Loss function evaluated every 10th iteration.
    """
    def __init__(self, C=5, Tau=1e-5, max_ite = 500, calc_loss=False):
        self.C = C
        self.Tau = Tau
        self.max_ite = max_ite
        self.calc_loss = calc_loss

    def fit(self, X, y):
        """Fit the model according to given training data. X's features are
        normalized to encourage stability.

        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            Training vector.

        y : np.array, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.fit_time = time.time()
        alpha = np.zeros(X.shape[0])

        Q = X.dot(X.T)
        Q /= Q.sum(axis=1)[:, np.newaxis]

        grad = np.zeros(X.shape[0])

        if self.calc_loss:
            loss = []

        i = j = ite = 0
        while (i != -1) and (j != -1) and (ite < self.max_ite):

            i,j  = self._wss(Q,y,grad,alpha)

            B = [i,j]
            #N = [l for l in range(X.shape[0]) if l not in B]
            #alpha_b, alpha_n = alpha[B], alpha[N]
            alpha_b = alpha[B]

            yi_ba_ij = y[i]*self._b(grad,y,i,j)**2 / self._a(Q,i,j)
            alpha[i] += yi_ba_ij
            alpha[j] -= yi_ba_ij

            #if y[i] != y[j]:

            if alpha[i] < 0:
                alpha[i] = 0
            else:
                alpha[i] = min(alpha[i], self.C)
            if alpha[j] < 0:
                alpha[j] = 0
            else:
                alpha[j] = min(alpha[j], self.C)


            grad += Q[:, B].dot(alpha[B] - alpha_b)


            if self.calc_loss:
                if ite%10 == 0:
                    loss.append(self._loss(Q,alpha))
                if ite >= 20:
                    if abs(loss[-2] - loss[-1]) < 1e-4:
                        break

            ite += 1

        if self.calc_loss:
            self.training_loss = loss

        self.fit_time = time.time() - self.fit_time
        self.alpha = alpha


    def _wss(self, Q, y, grad, alpha):
        """Active set selection. Search for a pair that approximately minizes
        the function value -b_it**2 / a_it.

        Parameters
        ----------
        Q : np.array, shape (n_samples, n_samples)
            Kernel matrix where each element is x[i,:].dot(x[i,:].T)

        y : np.array, shape(n_samples,)
            Target vector relative to X.

        alpha : np.array, shape(n_samples)
            Dual optimal variable.

        Returns
        ------
        (i, j) : tuple of ints
            Indices that approximiately minimize the function value.
        """

        I_up, I_low = self._update_i_up_low(y, alpha)

        # search for i
        i, i_max_f = -1, -np.infty
        for i_up in I_up:
            yt_grad = -y[i_up]*grad[i_up]
            if yt_grad > i_max_f:
                i, i_max_f = i_up, yt_grad

        # search for j
        j, j_min_f = -1, np.infty
        for j_up in I_low:
            ba_ij = -self._b(grad,y,i,j_up)**2 / self._a(Q,i,j_up)
            if (ba_ij < j_min_f) and (-y[j]*grad[j] < i_max_f + self.Tau):
                j, j_min_f = j_up, ba_ij

        return i, j


    def _a(self, Q, t, s):
        return max(self.Tau, Q[t,t] + Q[s,s] - 2*Q[t,s])


    def _b(self, grad, y, t, s):
        return max(self.Tau, -y[t]*grad[t] + y[s]*grad[s])


    def _update_i_up_low(self, y, alpha):
        I_up  = [t for t,at in enumerate(alpha)
                if (at < self.C and y[t] ==  1)
                or (at > 0      and y[t] == -1)]

        I_low = [t for t,at in enumerate(alpha)
                if (at < self.C and y[t] == -1)
                or (at > 0      and y[t] ==  1)]

        return I_up, I_low

    def _loss(self, Q, alpha):
        return 0.5 * alpha.dot(Q).dot(alpha) - sum(alpha)


    def score(self, X, y):
        return


def main():

    import matplotlib.pyplot as plt
    from time import localtime, strftime

    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    smo = SMO(calc_loss=True)
    smo.fit(X,y)

    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(smo.training_loss)
    plt.savefig('../plots/smo/loss_vs_ite_{}.pdf'.format(
        strftime("%Y.%m.%d_%H.%M.%S", localtime()),
        format='pdf'))
    plt.close('all')
    print('Loss vs Iteration saved to ../plots/smo/')
    print('Time: {} in {} iterations'.format(smo.fit_time,
        len(smo.training_loss)))
    print('Final objective function val: {}\n'.format(
        round(smo.training_loss[-1], 5)))


if __name__ == '__main__':
    main()
