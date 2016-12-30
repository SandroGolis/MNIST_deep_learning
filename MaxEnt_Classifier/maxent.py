# -*- mode: Python; coding: utf-8 -*-
from __future__ import division
from random import seed, shuffle
import numpy as np
import scipy.misc as scp

# Global parameters
from classifier import Classifier

LEARNING_RATE = 0.001  # learning rate for SGD
BATCH_SIZE = 100  # mini batch size
MAX_EPOCHS = 100  # max number of epochs
EARLY_STOPPING_STEP_SIZE = 5  # checking convergence every STEP_SIZE epochs
BIAS_FEATURE = '_demo_feature_for_bias_'


class MaxEntModel(object):
    """
          This class will contain all needed information for MaxEnt model.
          1. map_name_to_id       type: Dictionary(String->Integer)
                                        feature_name -> feature_id
            feature_id is the row number of corresponding feature in
            weights_matrix 2d array.
          2. map_label_to_col     type: Dictionary(String->Integer)
                                        class_name -> class_column
            class_column is the column number of corresponding label
            in weights_matrix 2d array.
          3. weights_matrix       type: numpy.array(rows, columns)
                        class_A     class_B     . . .      class_X
                      |----------|-----------|----------|----------|
               f_id_1 |   w1_A   |   w1_B    |          |   w1_X   |
               f_id_2 |   w2_A   |   w2_B    |          |   w2_X   |
                 .    |          |           |          |          |
                 .    |          |           |          |          |
                 .    |          |           |          |          |
               f_id_N |   wN_A   |   wN_B    |          |   wN_X   |
                      |----------|-----------|----------|----------|

            For example: if map_name_to_id[feature1] = f_id_1 then
                         w1_A is a weight corresponding to feature
               f_1A(X, C) = 1 if C=class_A AND feature1 is in features(X)
                            0 otherwise
            4. observed count     type: numpy.array(rows, columns)
               matrix with the same shape as weights_matrix.
               The values will be empirical counts of features in the training set.
    """

    def __init__(self, map_name_to_id, labels):
        super(MaxEntModel, self).__init__()
        self.map_label_to_col = labels
        self.map_name_to_id = map_name_to_id
        self.rows = len(map_name_to_id)
        self.cols = len(labels)
        self.weights_matrix = np.zeros((self.rows, self.cols))
        self.observed_count = None

    def update_weights(self, weights):
        self.weights_matrix = np.copy(weights)


def generate_batches(data, batch_size):
    """
        This generator yields slices of data.
        Each slice is a view on the initial data and it's size equals to batch_size
        if data doesn't divide evenly by mini batches, the last slice will be shorter
    """
    start_index = 0
    while True:
        last_index = start_index + batch_size
        yield data[start_index: last_index]
        start_index = last_index
        if start_index > data.size:
            return


def get_sparse_vector(fnames, map_name_to_id):
    names_set = set(fnames)
    sparse_vector = [map_name_to_id[BIAS_FEATURE]]
    for name in names_set:
        if name in map_name_to_id:
            sparse_vector.append(map_name_to_id[name])
    return sparse_vector


class MaxEnt(Classifier):
    def __init__(self, new_model=None):
        super(MaxEnt, self).__init__(new_model)

    def train(self, instances, dev_instances=None):
        """
                Construct a statistical model from labeled instances.
                I will build 2 data structures
                1. model.map_name_to_id (type: dictionary)
                   mapping: feature_name ---> feature_id
                   I will add a feature: BIAS_FEATURE to each data instance
                   in order to represent bias term in the weights matrix.
                2. model.map_label_to_col (type: dictionary)
                   mapping: class_name ---> class_column

                :type instances: list(Documents)
                :param instances: Training data, represented as a list of Documents

                :type dev_instances: list(Documents)
                :param dev_instances: Dev data, represented as a list of Documents
        """
        map_name_to_id = {BIAS_FEATURE: 0}
        labels = {}
        for instance in instances:

            # store mapping of each unseen label to a new column number
            label = instance.label
            if label not in labels:
                labels[label] = len(labels)

            # store mapping of each unseen feature to a new feature id
            fnames = instance.features()
            for fname in fnames:
                if fname not in map_name_to_id:
                    map_name_to_id[fname] = len(map_name_to_id)

            # Cache sparse feature_vector for train_instances
            instance.feature_vector = get_sparse_vector(fnames, map_name_to_id)

        # Initialize model data structures
        self.model = MaxEntModel(map_name_to_id, labels)

        # Cache sparse feature_vector for dew_instances
        for dev in dev_instances:
            dev.feature_vector = get_sparse_vector(dev.features(), map_name_to_id)

        # Calculate weights using Mini-batch Gradient Descent
        self.train_sgd(instances, dev_instances)

    # noinspection PyCompatibility
    def train_sgd(self, train_instances, dev_instances):
        """
            Train MaxEnt model with Mini-batch Gradient Descent.
            Using global parameters LEARNING_RATE and BATCH_SIZE
            Early stopping method is used every EARLY_STOPPING_STEP_SIZE iterations
            to prevent overfitting.

            As an optimization, generate_batches generator works on numpy.array instead of list
            to prevent unnecessary copying of slices. With numpy.array each batch is a view on the initial data.
        """
        last_accuracy = 0
        weights = self.model.weights_matrix
        for epoch in xrange(MAX_EPOCHS):
            seed(hash("shuffle_for_epoch" + str(epoch)))
            shuffle(train_instances)
            all_data = np.array(train_instances)

            for mini_batch in generate_batches(all_data, BATCH_SIZE):
                gradient = self.evaluate_gradient(mini_batch, weights)
                weights += LEARNING_RATE * gradient

            if epoch % EARLY_STOPPING_STEP_SIZE == 0:
                test_accuracy = self.get_accuracy(dev_instances, weights)
                if test_accuracy < last_accuracy:
                    break
                else:
                    last_accuracy = test_accuracy
                    self.model.update_weights(weights)

    def classify(self, instance, feature_vector=None, weights=None, return_distribution=False):
        sparse_vector = feature_vector
        if sparse_vector is None:
            fnames = instance.features()
            sparse_vector = get_sparse_vector(fnames, self.model.map_name_to_id)

        prob_dict = {}
        for label in self.model.map_label_to_col:
            prob = self.get_prob_given_X(label, sparse_vector, weights)
            prob_dict[label] = prob

        if return_distribution:
            return prob_dict
        else:
            return max(prob_dict, key=prob_dict.get)

    def get_prob_given_X(self, label, sparse_vector, weights=None):
        """
            Given a sparse_vector that corresponds to X
            calculates posterior probability P(Y=label | X)

            Using current model.weights_matrix as default.
            To avoid underflow while dividing 2 relatively close big numbers,
            I will calculate it in log space and take exponent in the end.
            Also using scipy.logsumexp function which is computationally stable
        """
        col = self.model.map_label_to_col[label]
        if weights is None:
            weights = self.model.weights_matrix
        # from weights_matrix get only the rows that correspond to the sparse_vector
        relevant_weights = weights[sparse_vector, :]
        weight_sums = relevant_weights.sum(axis=0)
        tmp = weight_sums[col] - scp.logsumexp(weight_sums)
        return np.exp(tmp)

    def calc_empirical_count(self, train_instances):
        observed_counts = np.zeros((self.model.rows, self.model.cols))
        for instance in train_instances:
            sparse_vector = instance.feature_vector
            col = self.model.map_label_to_col[instance.label]
            observed_counts[sparse_vector, col] += 1

        return observed_counts

    def calc_expected_count(self, train_instances, weights):
        """
            Count the probabilities of current model
            1. For each document
            2. For each class
                compute the probability P(y=class | document)
                and add to output matrix
        """
        expected_counts = np.zeros((self.model.rows, self.model.cols))
        for instance in train_instances:
            sparse_vector = instance.feature_vector
            for label, col in self.model.map_label_to_col.items():
                prob = self.get_prob_given_X(label, sparse_vector, weights)
                expected_counts[sparse_vector, col] += prob

        return expected_counts

    def get_accuracy(self, dev_instances, weights):
        if len(dev_instances) == 0:
            return 0
        count = 0
        for doc in dev_instances:
            if self.classify(doc, feature_vector=doc.feature_vector, weights=weights) == doc.label:
                count += 1
        return count / len(dev_instances)

    def evaluate_gradient(self, mini_batch, weights):
        observed_count = self.calc_empirical_count(mini_batch)
        expected_count = self.calc_expected_count(mini_batch, weights)
        return observed_count - expected_count

    def get_model(self):
        return self._model

    def set_model(self, new_model):
        self._model = new_model

    model = property(get_model, set_model)

