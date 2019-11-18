from nn.activation import MSE
from nn.layer import Layer
from typing import List
import numpy as np
import random


class SequentialNetwork:
    def __init__(self, loss=None):
        print("Initialize Network..")
        self.layers: List[Layer] = []
        if loss is None:
            self.loss = MSE()

    def add(self, layer: Layer):
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            print("Starting Epoch {}".format(epoch))
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            # count = 0
            for mini_batch in mini_batches:
                # count += 1
                # if count % 50 == 0:
                #     print("Mini batch: {} of {}".format(
                #         count, len(mini_batches)))
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)

        self.update(mini_batch, learning_rate)

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()
            last_layer = self.layers[-1]
            last_layer.input_delta = self.loss.loss_derivative(
                last_layer.output_data, y)
            for layer in reversed(self.layers):
                layer.backward()

    def single_forward(self, x):
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        test_results = [(
            np.argmax(self.single_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)  # <1>
        for layer in self.layers:
            layer.update_params(learning_rate)  # <2>
        for layer in self.layers:
            layer.clear_deltas()  # <3>
