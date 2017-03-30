import numpy as np
import numbers
import time

class LogisticRegression:
    def __init__(self, strategy='GD', lam=0.1, tol=1e-5, max_iter=1000):
        self.strategy = strategy
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        # TODO(pmdaly): update the additional optimization strategies
        # TODO(pmdaly): do class weights need to be added for multi-class?
        self.fit_time = time.time()
        self.classes = np.unique(y)
        self.n_classes = len(np.unique(y))
        if self.n_classes > 2:
            self.w = np.zeros((X.shape[1], len(self.classes)))
            for class_i in self.classes:
                y_class_i = np.where(y == class_i, 1, 0)
                self.w[:, class_i] = self.optimize(X, y_class_i)
            self.fit_time = time.time() - self.fit_time
        else:
            self.w = self.optimize(X, y)
            self.fit_time = time.time() - self.fit_time

    def optimize(self, X, y):
        # TODO(pmdaly): Gradient Descent is the only strategy that works atm
        if self.strategy == 'AGD':
            return self._agd(X,y)
        elif self.strategy == 'SGD':
            return self._sgd(X,y)
        elif self.strategy == 'SGD Batch':
            return self._sgd_batch(X,y)
        elif self.strategy == 'IRLS':
            return self._newton(X, y)
        else:
            return self._gradient_descent(X,y)


    def score(self, X, y):
        y_pred = np.zeros(y.shape, dtype=np.int)
        if self.n_classes > 2:
            for i in range(X.shape[0]):
                probas = [self._sigmoid(X[i,:], self.w[:,j])
                          for j in range(self.n_classes)]
                class_estimate = probas.index(max(probas))
                y_pred[i] = class_estimate
            return np.mean(y_pred == y)
        else:
            for i in range(X.shape[0]):
                proba = self._sigmoid(X[i,:], self.w)
                if proba < 0.5:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            return np.mean(y_pred == y)

    def _loss(self, X, y, w):
        loss = 0
        m = X.shape[0]
        for i in range(m):
            loss += -(y[i]*X[i,:].dot(w) - np.log(1 + np.exp(X[i,:].dot(w))))
        return loss / m

    def _sigmoid(self, X, w):
        return 1/(1 + np.exp(-X.dot(w)))

    def _gradient(self, X, y, w):
        return X.T.dot(self._sigmoid(X,w) - y)

    def _backtracking():
        # TODO(pmdaly): add backtracking to stablize optimizations
        return

    def _gradient_descent(self, X, y):
        m, n = X.shape
        w = np.random.uniform(0.1,1,n)
        alpha = 0.5
        prev_w = ite = 0

        while np.linalg.norm(prev_w - w)**2 > self.tol and ite < self.max_iter:
            #print('Loss: {:>15}, weight change: {:>15}'.format(
            #    np.round(self._loss(X,y,w), 2),
            #    np.round(np.linalg.norm(prev_w - w)**2), 5))
            prev_w = w
            w = w - alpha*self._gradient(X,y,w)
            ite += 1
        return w

    def _sgd(self, X, y):
        m, n = X.shape
        w = np.random.uniform(0.1,1,n)
        alpha = 0.1
        prev_w = ite = 0

        while np.linalg.norm(prev_w - w) > self.tol and ite < self.max_iter:
            prev_w = w
            i = np.random.randint(0,m)
            w = w - alpha*self._gradient(X[i,:],y[i],w)
            ite += 1
        return w

    def _sgd_batch(self, X, y, batch_size=10):
        m, n = X.shape
        w = np.random.uniform(0.1,1,n)
        alpha = 0.1
        prev_w = ite = 0

        while np.linalg.norm(prev_w - w) > self.tol and ite < self.max_iter:
            prev_w = w
            batch = np.random.randint(0,m, batch_size)
            w = w - alpha*self._gradient(X[batch,:],y[batch],w)
            ite += 1
        return w

    def _agd(self, X, y):
        # TODO(pmdaly): fix this, didnt' have time
        m, n = X.shape
        w = np.random.uniform(0.1,1,n)
        beta = 2
        z = [0]
        l = [0]*2
        prev_w = ite = 0
        while np.linalg.norm(prev_w - w)**2 > self.tol and ite < self.max_iter:
            prev_w = w
            z.append(w - self._gradient(X,y,w)/beta)
            l.append((1 + np.sqrt(1+4*l[-1]**2))/2)
            w = z[-1] + (l[-3]/l[-1])*(z[-1] - z[-2])
        return w

    def _irls(self, X, y):
        # TODO(pmdaly): doesn't work either
        m, n = X.shape
        w = np.random.uniform(0.1,1,n)
        z = np.random.uniform(0.1,1,m)

        def hessian(w, y):
            A = np.zeros((m,m))
            for i in range(m):
                A[i,i] = self._sigmoid(X[i,:],w)*(1 - self._sigmoid(X[i,:],w))
                z[i] = X[i,:].dot(w) + (1-self._sigmoid(y[i]*X[i,:],w))*y[i]/A[i,i]
            return (X.T.dot(A).dot(X)) + self.lam * np.identity(n), A, z

        ite = 0
        prev_w = 0
        while np.linalg.norm(prev_w - w)**2 > self.tol and ite < self.max_iter:
            #print(np.linalg.norm(prev_w - w)**2)
            #print(self._loss(X,y,w))
            #print('Loss: {:>15}, weight change: {:>15}'.format(
            #    np.round(self._loss(X,y,w), 2),
            #    np.round(np.linalg.norm(prev_w - w)**2), 5))
            prev_w = w
            H, A, z = hessian(w, y)
            w = np.linalg.inv(-H).dot(X.T).dot(A).dot(z)
            ite += 1
        return w


def _check_init_option(max_iter, tol):
    if not isinstance(max_iter, numbers.Number) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be positive;"
                            " got (max_iter={})".format(max_iter))
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                            "positive; got (tol={})".format(tol))

def main():
    from sklearn.linear_model import LogisticRegression as SKLR
    from sklearn.datasets import load_digits
    Digits = load_digits()
    X, y = Digits['data'], Digits['target']
    logit = LogisticRegression()
    logit.fit(X,y)
    logit_sk = SKLR().fit(X,y)
    print('\nSK accuracy: {} vs. Our accuracy: {}'.format(
            logit_sk.score(X,y), logit.score(X,y)))

if __name__ == '__main__':
    main()
