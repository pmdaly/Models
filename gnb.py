import numpy as np

class GaussianNaiveBayes:
    #
    def __init__(self, tol=1e-5, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
    #
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        m, n = X.shape
        self._set_priors(X.shape[0], y)
        self._set_class_means(X,y)
        self._set_covariance(X)
    #
    def _set_priors(self, m,y):
        self.priors = np.array(np.bincount(y) / m)
    #
    def _set_covariance(self, X):
        #Cov = np.diagflat(np.diag(np.cov(X.T)))
        Cov = np.cov(X.T)
        if not self._is_invertible(Cov):
            Cov += np.identity(Cov.shape[0])*1e-8
        self.covariance = Cov
    #
    def _set_class_means(self, X, y):
        self.class_means = np.zeros((self.n_classes, X.shape[1]))
        for c in self.classes:
            self.class_means[c] = np.mean(X[np.where(y==c)], axis=0)
    #
    def score(self, X, y):
        y_pred = np.zeros(len(y), dtype=np.int)
        for i,x in enumerate(X):
            max_posterior = (0,-1)
            exp_ak = self._compute_exp_ak(x)
            exp_ak /= sum(exp_ak)
            y_pred[i] = np.argmax(exp_ak)
        return np.mean(y_pred == y)
    #
    def _is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    #
    def _compute_exp_ak(self, x):
        exp_ak = np.zeros(self.n_classes)
        for c in self.classes:
            mk = self.class_means[c]
            wk = np.linalg.inv(self.covariance).dot(mk)
            wk0 = -0.5*mk.T.dot(np.linalg.inv(self.covariance)).dot(mk)
            exp_ak[c] = np.exp(wk.T.dot(x) + wk0)
        return exp_ak

def main():
    from cross_validation import train_test_split
    from sklearn.datasets import load_digits
    Digits = load_digits()
    X, y = Digits['data'], Digits['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train,y_train)
    error = gnb.score(X_test,y_test)
    print(error)

if __name__ == '__main__':
    main()
