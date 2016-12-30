from __future__ import print_function
import time
import theano
import theano.tensor as T
import lasagne
import models
import numpy as np
from load_dataset import load_dataset

# Global parameters
LEARNING_RATE = 0.01  # learning rate for SGD
MOMENTUM = 0.9  # Nesterov momentum parameter for SGD
BATCH_SIZE = 500  # mini batch size
MAX_EPOCHS = 200  # max number of epochs
EARLY_STOPPING_STEP_SIZE = 10  # checking convergence every STEP_SIZE epochs
LAYER_WIDTH = 800  # number of units in each layer
DROP_INPUT_RATE = 0.2  # 20% random dropout to input layer
DROPP_HIDDEN_RATE = 0.5  # 50% random dropout to hidden layer


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

user_answer = raw_input("Choose from the following:\n"
                        "1 - Neural network with 1 hidden layer\n"
                        "2 - Neural network with 2 hidden layers\n"
                        "3 - Convolutional neural network with 2 stages\n")
if user_answer == "1":
    print("Building MPL model with 1 hidden layer")
    network = models.build_custom_mlp(input_var, 1, LAYER_WIDTH, DROP_INPUT_RATE, DROPP_HIDDEN_RATE)
elif user_answer == "2":
    print("Building MPL model with 2 hidden layers")
    network = models.build_custom_mlp(input_var, 2, LAYER_WIDTH, DROP_INPUT_RATE, DROPP_HIDDEN_RATE)
elif user_answer == "3":
    print("Building CNN model with 2 stages")
    network = models.build_cnn(input_var)
else:
    print("Invalid option. Exiting.")
    exit()

# Load the dataset. Will be downloaded if doesn't exist
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Create a loss expression for training, i.e., a scalar objective we want to minimize
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Create update expressions using Stochastic Gradient Descent (SGD)
# Nesterov momentum is used (MOMENTUM)
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=LEARNING_RATE, momentum=MOMENTUM)

# Create a loss expression for validation/testing.
# A deterministic forward pass through the network, disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

# Expression for the classification accuracy
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a function computing the validation loss and accuracy
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Training loop
print("Starting training...")
last_accuracy = 0
for epoch in xrange(MAX_EPOCHS):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    current_accuracy = val_acc / val_batches * 100
    if epoch % EARLY_STOPPING_STEP_SIZE == 0:
        if current_accuracy < last_accuracy:
            break
        else:
            last_accuracy = current_accuracy

    # Epoch results
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, MAX_EPOCHS, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(current_accuracy))

# Evaluate on test set
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
