import numpy as np

class Perceptron:

    def __init__(self, eta=0.1, epochs=100):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):

        # 1 represents the bias
        self.w = np.zeros(1 + X.shape[1])

        # this should probably loss
        self.accuracy = []

        X = np.append(np.ones((X.shape[0],1)), X, 1)

        for _ in range(epochs):

