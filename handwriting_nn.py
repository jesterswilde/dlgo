from nn.sequential_network import SequentialNetwork
from nn.activation_layer import Activation_Layer
from nn.dense_layer import DenseLayer
from handwriting.load_writing import load_data
import numpy as np

training_data, test_data = load_data()

print("Training Data: ")

net = SequentialNetwork()

net.add(DenseLayer(784, 392))
net.add(Activation_Layer(392))
net.add(DenseLayer(392, 198))
net.add(Activation_Layer(198))
net.add(DenseLayer(198, 10))
net.add(Activation_Layer(10))

net.train(training_data, epochs=10, mini_batch_size=10,
          learning_rate=3.0, test_data=test_data)
