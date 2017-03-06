import numpy as np
from functools import reduce

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

        if self.hidden_layers > 1:
            self.W = [np.zeros((X.shape[1], X.shape[1]))
                      for _ in range(self.hidden_layers - 1),
                      np.zeros((X.shape[1], 1))]
        else:
            self.W = np.zeros((1 + X.shape[1]))
        self.loss = []

        X = np.append(np.ones((X.shape[0],1)), X, 1)
        y = np.array([-1 if yi == 0 else 1 for yi in y])

        for _ in range(self.epochs):
            y_ = self._forward(X, y)

    def _forward(self, X, y):
        return self._sigmoid(self._chain_mult([X,*self.W]))

    def _backward(self, X, y):

    def score(self, X, y):

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _chain_mult(args):
        return reduce(np.dot, args)


def main():

if __name__ == '__main__':
    main()
