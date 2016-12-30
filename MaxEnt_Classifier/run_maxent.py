import sys
import os
import numpy as np

from MaxEnt_Classifier.maxent import MaxEnt
from load_dataset import load_dataset
from corpus import Document


class NLTK_image(Document):
    def features(self):
        # return a list of indices that have non zero values.
        # it is flatnonzero, so the indices are taken from an array of size 28*28=724
        return np.flatnonzero(self.data)

class NLTK_corpus(object):
    """A corpus in which each document is an NLTK image."""

    def __init__(self, data, labels):
        super(NLTK_corpus, self).__init__()
        self.documents = []

        num_images,_,_,_ = data.shape
        for i in xrange(num_images):
            image = NLTK_image(data[i], labels[i])
            self.documents.append(image)

    def __len__(self): return len(self.documents)

    def __iter__(self): return iter(self.documents)

    def __getitem__(self, key): return self.documents[key]

    def __setitem__(self, key, value): self.documents[key] = value

    def __delitem__(self, key): del self.documents[key]


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(os.path.pardir)

train = NLTK_corpus(X_train, y_train)
dev = NLTK_corpus(X_val, y_val)
test = NLTK_corpus(X_test, y_test)

classifier = MaxEnt()
classifier.train(train, dev)
accuracy(classifier, test)