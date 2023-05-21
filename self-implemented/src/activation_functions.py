import numpy as np


class Nothing:
    def get_function(self):
        return lambda x: x

    def get_derivative(self):
        return lambda x: x


class ReLU:
    def get_function(self):
        return lambda x: np.maximum(0, x)

    def get_derivative(self):
        """
        if x > 0:
            y = 1
        elif x <= 0:
            y = 0
        """
        return lambda x: np.greater(x, 0).astype(int)


class Sigmoid:
    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        fx = self.activate(x)
        return fx * (1.0 - fx)
