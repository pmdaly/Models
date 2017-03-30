'''
General structure

1. Compute d-dimensional mean vectors for each class
2. Compute scatter matrices (w/in class Sw and between class Sb)
3. Compute eig vec, value pairs
4. Sort eig values by decreasing value or use svd
5. Y = XW

TODO(pmdaly): #'s are carriage return placeholders b/c screen send doesn't
    work, remove later
'''

import numpy as np
import time

class LinearDiscriminantAnalysis:

    def __init__(self, tol=1e-5, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y, dimensions=1):
        self.classes = np.unique(y)
        self.n_classes = len(np.unique(y))
        self.w = np.zeros((X.shape[1], len(self.classes)))
        Means = self._compute_class_mean_vectors(X,y)
        Sw, Sb = self._compute_scatter_matrices(X,y,Means)
        Eigs = self._get_sorted_eigs(Sw, Sb)
        self.w = np.real(np.column_stack(Eigs[d][1] for d in range(dimensions)))
        self._set_projected_means(X,y)

    def _set_projected_means(self, X, y):
        P = X.dot(self.w)
        self.projected_means = []
        for c in self.classes:
            self.projected_means.append(np.mean(P[np.where(y==c)]))

    def _get_sorted_eigs(self, Sw, Sb):
        Lam, V = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        Eigs = list(zip(Lam,[V[:,col] for col in range(V.shape[1])]))
        Eigs.sort(key=(lambda x: x[0]), reverse=True)
        return Eigs

    def _compute_scatter_matrices(self, X, y, Means):
        n = X.shape[1]
        Sw = np.zeros((n,n))
        for k in range(self.n_classes):
            class_k_idx = np.where(y == k)[0]
            for j in class_k_idx:
                Sw += np.outer(X[j,:] - Means[k,:],X[j,:] - Means[k,:])
        Sb = np.zeros((n,n))
        X_mean = np.mean(X, axis=0)
        class_counts = np.bincount(y)
        for k in range(self.n_classes):
            Ni = class_counts[k]
            Sb += Ni*np.outer(Means[k,:]-X_mean,Means[k,:]-X_mean)
        if not self._is_invertible(Sw):
            Sw += np.identity(Sw.shape[0])*1e-8
        return Sw, Sb

    def _compute_class_mean_vectors(self, X, y):
        Means = np.zeros((self.n_classes, X.shape[1]))
        for i in self.classes:
            class_i_idx = np.where(y == i)[0]
            Means[i,:] = np.mean(X[class_i_idx,:], axis=0)
        return Means

    def _is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def score_lda_1d(lda, X, y):
    # TODO(pmdaly): change this so there is a fitting on train and scoring on
    Projection = np.squeeze(X.dot(lda.w))
    c0_mean = lda.projected_means[0]
    c1_mean = lda.projected_means[0]
    decision = (c0_mean + c1_mean) / 2
    if c1_mean > c0_mean:
        y_pred = np.where(Projection > decision, 1, 0)
    else:
        y_pred = np.where(Projection > decision, 0, 1)
    return np.mean(y_pred == y)

def score_lda_2d(lda, X, y):
    Projection = np.squeeze(X.dot(lda.w))
    class_means = lda.projected_means
    y_pred = [0]*len(y)
    for p_index, p in enumerate(Projection):
        dist = (np.linalg.norm(p), -1)
        for c_index, c_mean in enumerate(class_means):
            p_dist = np.linalg.norm(p-c_mean)
            if p_dist < dist[0]:
                dist = (p_dist, c_index)
        y_pred[p_index] = dist[1]
    return np.mean(y_pred == y)

def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDSK
    #from sklearn.datasets import load_digits
    #Digits = load_digits()
    from sklearn.datasets import load_boston
    Boston = load_boston
    #X, y = Digits['data'], Digits['target']
    X, y = Boston['data'], Boston['target']
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    lda.score(X,y)
    lda_sk = LDSK().fit(X,y)
    print('\nSK accuracy: {} vs. Our accuracy: {}'.format(
            lda_sk.score(X,y), lda.score(X,y)))

if __name__ == '__main__':
    main()
