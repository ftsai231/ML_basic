# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random


def dist(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        self.centroids = X_train[np.random.choice(len(X_train), centers)]

        iteration = 0
        prev_centroids = None

        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]

            # first loop, assign each data point to the closet centroid
            for x in X_train:
                dists = dist(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids

            # second loop, update centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]

    def eval(self, X):
        centroids = []
        centroids_idxs = []
        for x in X:
            dists = dist(x, self.centroids)
            centroids_idx = np.argmin(dists)
            centroids.append(self.centroids[centroids_idx])
            centroids_idxs.append(centroids_idx)

        return centroids, centroids_idxs


centers = 5
X_train, true_labels = make_blobs(n_samples=100, centers=centers)
X_train = StandardScaler().fit_transform(X_train)
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

class_centers, classification = kmeans.eval(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification)

plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )
plt.show()











