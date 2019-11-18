import numpy as np
import gzip
import six.moves.cPickle as pickle
import os


def encode_label(label):
    output = np.zeros((10, 1))
    output[label] = 1.0
    return output


def reshape_data(data):
    reshaped = [np.reshape(x, (784, 1)) for x in data[0]]

    labels = [encode_label(y) for y in data[1]]

    return list(zip(reshaped, labels))


def load_data_impl():
    # file retrieved by:
    #   wget https://s3.amazonaws.com/img-datasets/mnist.npz -O code/dlgo/nn/mnist.npz
    # code based on:
    #   site-packages/keras/datasets/mnist.py
    path = os.path.dirname(__file__)+'/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def load_data():
    train_data, test_data = load_data_impl()
    return reshape_data(train_data), reshape_data(test_data)
