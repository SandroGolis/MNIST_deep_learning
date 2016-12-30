# -*- mode: Python; coding: utf-8 -*-

"""A simple framework for supervised text classification."""

from abc import ABCMeta, abstractmethod, abstractproperty
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL


class Classifier(object):
    """An abstract text classifier.

    Subclasses must provide training and classification methods, as well as
    an implementation of the model property. The internal representation of
    a classifier's model is entirely up to the subclass, but the read/write
    model property must return/accept a single object (e.g., a list of
    probability distributions)."""

    __metaclass__ = ABCMeta

    def __init__(self, new_model=None):
        if isinstance(new_model, (basestring, file)):
            self.load_model(new_model)
        else:
            self.model = new_model

    def get_model(self): return None

    def set_model(self, model): pass

    model = abstractproperty(get_model, set_model)

    def save(self, my_file):
        """Save the current model to the given file."""
        if isinstance(my_file, basestring):
            with open(my_file, "wb") as my_file:
                self.save(my_file)
        else:
            dump(self.model, my_file, HIGHEST_PICKLE_PROTOCOL)

    def load(self, my_file):
        """Load a saved model from the given file."""
        if isinstance(my_file, basestring):
            with open(my_file, "rb") as my_file:
                self.load(my_file)
        else:
            self.model = load(my_file)

    @abstractmethod
    def train(self, instances):
        """Construct a statistical model from labeled instances."""
        pass

    @abstractmethod
    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None
