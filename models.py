from __future__ import print_function
import lasagne


# ##################### Build the neural network model #######################
# 3 types of models.
# Input: Theano variable representing the input
# Output: The output layer of a neural network model built in Lasagne.
#
# 1. build_mlp_2 - Multi-Layer Perceptron of 2 hidden layers of 800 units each
#                  followed by a softmax output layer of 10 units.
#                  It applies 20% dropout to the input data and 50% dropout to the hidden layers.
#
# 2. build_custom_mlp - is a generalized build_mlp. Can build a custom multi-layer network
#                       with or without dropouts
#
# 3. build_cnn - Convolutional Neural Network of two convolution + pooling stages
#                and a fully-connected hidden layer in front of the output layer.


def build_mlp_2(input_var=None):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    # Apply 20% dropout to the input data
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected hidden layer of 800 units.
    # linear rectifier is used as an activation function
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify)

    # Dropout of 50%
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another hidden layer of 800-units
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # fully-connected Softmax output layer
    l_out = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
            network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Convolution layer with 32 kernels of size 5x5
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network
