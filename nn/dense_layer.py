from nn.layer import Layer
import numpy as np
from nptyping import Array


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight: Array[float] = np.random.randn(
            output_dim, input_dim)
        self.bias: Array[float] = np.random.randn(
            output_dim, 1)
        self.params = (self.weight, self.bias)

        self.delta_weight: Array = np.zeros(
            self.weight.shape)
        self.delta_bias: Array = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()
        self.delta_bias += delta
        self.delta_weight = np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)

    def update_params(self, learning_rate):
        self.weight -= learning_rate * self.delta_weight
        self.bias -= learning_rate * self.delta_bias

    def clear_deltas(self):
        self.delta_weight = np.zeros(self.weight.shape)
        self.delta_bias = np.zeros(self.bias.shape)

    def describe(self):
        print("|---  " + self.__class__.__name__)
        print("   |-- dimensions: ({} {})"
              .format(self.input_dim, self.output_dim))
