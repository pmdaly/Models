import numpy as np
from functools import reduce

import ipdb
#ipdb.set_trace()

class MLPClassifier:
    """Multilayer Perceptron Classifier

    Parameters
    ----------
    eta : float (default: 0.1), learning rate
    epochs : int (default: 100), number of passes over the dataset
    hidden_layers : int (default: 1), number of hidden perceptron layers

    Attributes
    ----------
    W : list, contains weights of each hidden layer as well as the final output
        layer
    loss : list, loss function evaluated after each epoch
    """

    def __init__(self, eta=0.1, epochs=100, n_layers=1):
        self.eta = eta
        self.epochs = epochs
        self.hidden_layers = n_layers

    def fit(self, X, y):

        X = np.append(np.ones((X.shape[0],1)), X, 1)
        y = np.array([-1 if yi == 0 else 1 for yi in y])

        self.W = [
                np.zeros((X.shape[1], X.shape[1]))
                for _ in range(self.hidden_layers - 1)
                ]
        self.W.append(np.zeros(X.shape[1]))


        self.layers = [X] + [self.W]
        self.loss = []

        for _ in range(self.epochs):
            y_ = self._forward(X, y)
            self._backprop(y)

    def _forward(self, X, y):
        for i in range(1, len(self.layers)):
            previous_layer = self.layers[i-1]
            self.layers[i] = self._sigmoid(
                    np.dot(previous_layer, self.W[i-1])
                    )
        return y - self.layers[-1]

    def _forward_old(self, X, y):
        '''Clean function that multiplies the input by the chain of weights, I
        can probably keep this for prediction
        '''
        f = lambda x, y : self._sigmoid(np.dot(x, y))
        return reduce(f, [X,*self.W])

    def _backprop(self, y):
        '''Either a dimensionality mismatch or wrong operation, i.e. inner
        product instead of Hadamard.
        '''

        layer_error = y - self.layers[-1]

        for i in range(len(self.layers)):
            layer_grad = layer_error * self._sigmoid(self.layers[i], deriv=True)
            self.W[i] += self.layers[i].T.dot(layer_grad)
            error = layer_grad.dot(self.W[i])

    def score(self, X, y):
        X = np.append(np.ones((X.shape[0],1)), X, 1)
        y = np.array([-1 if yi == 0 else 1 for yi in y])
        y_ = self._forward(X, y)
        return np.mean(y_ == y)

    def _sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1-x)
        return 1.0 / (1.0 + np.exp(-x))


def main():
    # need a better comparisson for MLP, this is just meant to say 'hey my MLP
    #   classification isn't terrible when compared to fully supported
    #   industrial classifiers'

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
    accuracy_mlp, accuracy_logit = [], []

    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        mlp, logit = MLPClassifier(), LogisticRegression()
        mlp.fit(X_train, y_train)
        logit.fit(X_train, y_train)
        accuracy_mlp.append(mlp.score(X_test, y_test))
        accuracy_logit.append(logit.score(X_test, y_test))

    print('Comparing Multilayer Perceptron to LogisticRegression as a baseline')
    print('MLP accuracy: {}, LR Accuracy: {}'.format(
        round(np.mean(accuracy_mlp),3), round(np.mean(accuracy_logit),3)))


if __name__ == '__main__':
    main()
