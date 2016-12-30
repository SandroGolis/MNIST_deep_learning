from urllib import urlretrieve
import gzip
import os
import numpy as np


def load_dataset(dest_path=""):
    """Downloads MNIST dataset and loads it into numpy arrays."""
    def download(filename, filepath, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filepath)

    def load_mnist_images(fname):
        file_path = os.path.join(dest_path, fname)
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            download(fname, file_path)

        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, reshaping them to 2D images
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, converting them to float32 in range [0,1].
        return data / np.float32(256)

    def load_mnist_labels(fname):
        file_path = os.path.join(dest_path, fname)
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            download(fname, file_path)

        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers
        return data

    # download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test