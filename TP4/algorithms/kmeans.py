import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style


# https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        # empty dictinary
        self.centroids = {}

        # shuffle data
        random.shuffle(data)

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # dictiory: key= centroids, value=sets contain within those values
            self.classifications = {}
            # it changes every time the centroid changes

            for i in range(self.k):
                self.classifications[i] = []

            for set in data:
                # creating a list that is being populated with k-number values. Distance to centroids
                distances = [np.linalg.norm(set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(set)

            # compare two centroids for tolerance value
            prev_centroids = dict(self.centroids)

            sum=0
            for classification in self.classifications:
                # centroid for all of the values. mean of all the features, and redefines the centroid
                pass
                # l = len(self.classifications)
                # self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # if any of the centroids moved more than the tolerance, then we are not optimized
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    print(abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)))
                    optimized = False
            # break the for loop when it made it with the tolerance
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11]])
#
# colors = 10 * ["g", "r", "c", "b", "k"]
#
# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()
#
# clf = K_Means()
# clf.fit(X)
#
# for centroid in clf.centroids:
#     plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
#                 marker="o", color="k", s=150, linewidth=5)
#
# for classification in clf.classifications:
#     color = colors[classification]
#     for featureset in clf.classifications[classification]:
#         plt.scatter(featureset[0], featureset[1], marker="x",
#                     color=color, s=150, linewidths=5)
#
# unknowns = np.array([[1, 3],
#                      [8, 9],
#                      [0, 3],
#                      [5, 4],
#                      [6, 4]])
# for unknown in unknowns:
#     classification = clf.predict(unknown)
#     plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
#
# plt.show()
