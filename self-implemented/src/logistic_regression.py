import time
import random
import numpy as np
from activation_functions import Sigmoid


class LogRegression:
    def __init__(self, epochs=5000, alpha=0.05):
        self.epochs = epochs
        self.activation_function = Sigmoid()
        self.alpha = alpha
        self.bias = 0

    def predict(self, x):
        return self.activation_function.activate(x @ self.weights + self.bias)

    def train(self, x, y, stochastic=False):
        start_time = time.time()

        self.x = np.array(x)
        self.y = np.array(y)
        self.n, self.in_size = self.x.shape

        self.weights = np.random.rand(self.in_size)
        self.bias = 0

        costs = {}
        for epoch in range(0, self.epochs):
            if stochastic:
                row = random.randint(0, self.n)
                x = self.x[row]
                y = self.y[row]
                diff = self.predict(x) - y
            else:
                x = self.x
                diff = self.predict(x) - self.y

            del_w, del_b = self._gradient(diff, x)
            self._update_parameters(del_w, del_b)

            if epoch % 1000 == 0:
                if stochastic:
                    cost = diff**2
                else:
                    cost = (diff**2).sum(axis=0) / self.n
                costs[epoch] = cost.sum()
                print(
                    f"Epoch: {epoch}, Cost: {costs[epoch]}, Time Spent: {time.time() - start_time:.2f}s"
                )
        return costs

    def _update_parameters(self, del_w, del_b):
        self.weights -= del_w * self.alpha
        self.bias -= del_b * self.alpha

    def _gradient(self, diff, x):
        del_w = (
            1
            / self.n
            * np.sum(
                2
                * x.T
                * diff
                * self.activation_function.derivative(x @ self.weights + self.bias)
            )
        )
        del_b = (
            1
            / self.n
            * np.sum(
                2
                * diff
                * self.activation_function.derivative(x @ self.weights + self.bias)
            )
        )

        return del_w, del_b
