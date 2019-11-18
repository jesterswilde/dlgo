from nn.layer import Layer
from nn.activation import sigmoid, sigmoid_derivative


class Activation_Layer(Layer):
    def __init__(self, input_dim):
        super(Activation_Layer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activation = sigmoid
        self.reverse_activation = sigmoid_derivative

    def forward(self):
        data = self.get_forward_input()
        self.output_data = self.activation(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * self.reverse_activation(data)

    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("   |-- dimensions: ({}, {}) "
              .format(self.input_dim, self.output_dim))
