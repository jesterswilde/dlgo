import numpy as np

__all__ = [
    'MSE',
    'sigmoid_double',
    'sigmoid',
    'sigmoid_derivative',
    'sigmoid_double_derivative'
    'predict'
]


class MSE():
    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels


def predict(inputs, weights, bias):
    return sigmoid_double(np.dot(inputs, weights) + bias)


# tag::sigmoid[]
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)
# end::sigmoid[]


def sigmoid_double_derivative(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_derivative(x):
    return np.vectorize(sigmoid_double_derivative)(x)
