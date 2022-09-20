# https://www.askpython.com/python/examples/k-nearest-neighbors-from-scratch

import numpy as np
from scipy.stats import mode

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from numpy.random import randint


def eucledian(p1, p2):
    return np.sqrt(np.sum(p1 - p2) ** 2)


def knn(x_train, y, x_input, k):

    output_labels = []

    for item in x_input:
        point_dist = []

        for j in range(len(x_train)):
            distances = eucledian(x_train[j, :], item)
            point_dist.append(distances)

        point_dist = np.array(point_dist)

        # return the index
        dist = np.argsort(point_dist)[:k]
        labels = y[dist]

        # majority voting
        lab = mode(labels)
        # output is something like this:
        # ModeResult(mode=array([0]), count=array([1]))
        lab = lab.mode[0]
        output_labels.append(lab)

    return output_labels


# The iris dataset is a classic and very easy multi-class classification dataset.
iris = load_iris()
X = iris.data
y = iris.target

train_idx = randint(0, 150, 100)
X_train = X[train_idx]
y_train = y[train_idx]

test_idx = randint(0, 150, 50)
X_test = X[test_idx]
y_test = y[test_idx]

y_pred = knn(X_train, y_train, X_test, 7)

print(y_pred)




