import numpy as np

class Perceptron:

    """Perceptron Classifier

    Parameters
    ----------
    eta : float (default: 0.1), learning rate
    epochs : int (default: 100), number of passes over the dataset

    Attributes
    ----------
    w : np.array, shape = (X.shape[1], 1)
    loss : list, loss function evaluated after each epoch
    """

    def __init__(self, eta=0.1, epochs=100)
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        """Fit the model according to given training data. A column of 1s is
        prepended to X for bias. y is assumed to be {0,1}.

        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            Training vector.

        y : np.array, shape(n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """

        self.w = np.zeros(1 + X.shape[1])
        self.loss = []

        X = np.append(np.ones((X.shape[0],1)), X, 1)
        y = np.array([-1 if yi == 0 else 1 for yi in y])

        for _ in range(self.epochs):
            for xi, yi in zip(X,y):
                yi_ = np.sign(np.dot(xi, self.w))
                self.w = self.w + self.eta * (yi-yi_)*xi

    def score(self, X, y):
        X = np.append(np.ones((X.shape[0],1)), X, 1)
        y = np.array([-1 if yi == 0 else 1 for yi in y])
        y_ = np.sign(X.dot(self.w))
        return np.mean(y_ == y)


def main():

    from sklearn.datasets import make_classification
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LogisticRegression

    # build a generic classifier
    X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=20,
            n_classes=2
            )
    accuracy_pcept, accuracy_logit = [], []

    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        pcept, logit = Perceptron(), LogisticRegression()
        pcept.fit(X_train, y_train)
        logit.fit(X_train, y_train)
        accuracy_pcept.append(pcept.score(X_test, y_test))
        accuracy_logit.append(logit.score(X_test, y_test))

    print('Comparing Perceptron to LogisticRegression as a baseline')
    print('Perceptron accuracy: {}, Logistic Regression Accuracy: {}'.format(
        round(np.mean(accuracy_pcept),3), round(np.mean(accuracy_logit),3)))


if __name__ == '__main__':
    main()
