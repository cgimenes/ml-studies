import numpy as np
import time
import random
from .activation_functions import Nothing


class FCLayer:
    def __init__(self, in_size, out_size, activation_function=Nothing()):
        self.in_size = in_size
        self.epochs = 5000
        self.weights = np.random.rand(self.in_size, out_size)
        self.biases = np.zeros(out_size)
        self.activation_function = activation_function.get_function()

    def run(self, x):
        return self.activation_function(x @ self.weights + self.biases)

    def update_parameters(self, cost, x, alpha=0.05):
        del_w, del_b = self._gradient(cost, x)
        self.weigth -= del_w * alpha
        self.bias -= del_b * alpha

    def _gradient(self, cost, x):
        del_w = 2 * np.sum(x * cost, axis=0)
        del_b = 2 * np.sum(cost)

        return del_w, del_b


class NeuralNetwork:
    def __init__(self, layers, epochs=50):
        self.epochs = epochs
        self.layers = layers

    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.run(input)
        return input

    def _forward_pass(self):
        cost = 0
        for i in range(0, self.n):
            x = self.x[i]
            y = self.y[i]
            diff = self.predict(x) - y
            cost += diff**2
        return cost / self.n

    def train(self, x, y, stochastic=True):
        start_time = time.time()

        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(self.x)

        costs = []
        for epoch in range(0, self.epochs):
            if stochastic:
                row = random.randint(0, self.n)
                x = self.x[row]
                y = self.y[row]
                diff = self.predict(x) - y
                cost = diff**2
            else:
                x = self.x
                cost = self._forward_pass()
            self._update_layers(cost, x)

            if epoch % 10 == 0:
                costs.append([epoch, cost])
                print(f"Epoch: {epoch}, Time Spent: {time.time() - start_time:.2f}s")
        return np.array(costs)

    def _update_layers(self, cost, x):
        for layer in self.layers:
            layer.update_parameters(cost, x)
