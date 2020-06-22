import numpy as np
import pandas as pd

import random
# this is a basic implementation fo k_means algorithm

class KMeans():
    def __init__(self, k=2, max_iter=10, tol = 0.001):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, train_data, train_labels):

        # {0, 1} - Enfermo y no enfermo
        labels = set(train_labels)
        self.space_dim = train_data.shape[1]
        self.centroids = []
        classification = []

        for entry in train_data:
            # le voy a asignar cualquier resultado, enfermo o no enfermo, a cada entrada del clasificador
            label = random.choice(list(labels))
            classification.append((entry, label))

        # interate util convergance or max iterations
        iteration = 0
        # until there is no condition satisfied, we keep going
        # if the two classifications are the same, we are done, otherwise we keep
        # on going until the amount of iterations are satisfied
        prev_classification = None
        while not self.converge(prev_classification, classification) and iteration < self.max_iter:
        # while not self._check_tol and iteration < self.max_iter:
            prev_classification = classification.copy()
            classification = []

            for label in labels:
                centroid = self.get_centroid(prev_classification, label)
                self.centroids.append((label, centroid))
            for v in train_data:
                new_label = self.distance(v, self.centroids)
                classification.append((v, new_label))
            iteration += 1


    def predict(self, data):
        predictions = []
        for entry in data:
            pred = self.distance(entry, self.centroids)
            predictions.append(pred)

        return np.array(predictions)

    # centroid is the mean value of all the observations/points
    # self.centroids[classification] = np.average(self.classifications[classification], axis=0)
    def get_centroid(self, prev_classification, label):

        centroid = np.zeros(self.space_dim)
        count = 0

        for entry, current_label in prev_classification:
            if current_label != label:
                break

            centroid += entry
            count += 1
        if count > 0:
            centroid /= count
        return centroid

    def distance(self, entry, centroids):
        # euclidean distance calculation
        # numpy norm is l2
        distances = []
        for label, point in centroids:
            distance = np.linalg.norm(entry - point)
            distances.append((distance, label))
        min_tuple = min(distances, key = lambda t: t[0])
        return min_tuple[1]

    def converge(self, prev_classification, new_classification):
        #check classifications

        if prev_classification == None or new_classification == None:
            return False

        for idx in range(len(prev_classification)):
            entry, prev_label = prev_classification[idx]
            entry, new_label = new_classification[idx]
            if prev_label != new_label:
                return False

        return True

    def _check_tol(self, prev_classification, new_classification):
        prev = 0
        new = 0
        for i in range(len(prev_classification)):
            prev += prev_classification[i][1]
        for i in range(len(new_classification)):
            new += new_classification[i][1]
        result = (new - prev) / len(prev_classification)

        if result < self.tol:
            return True

        return False


