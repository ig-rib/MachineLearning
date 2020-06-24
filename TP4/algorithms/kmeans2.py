import numpy as np
import pandas as pd

import random
# this is a basic implementation fo k_means algorithm

class KMeans():
    def __init__(self, k=2, max_iter=25000, tol = 0.00001):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, train_data, train_labels=None):

        # {0, 1} - Enfermo y no enfermo
        # labels = set(train_labels)
        # 1. Asignar aleatoriamente 1 a K a cada una de las obs
        # 2.
            # calcular centroide
            # asignar centroide al cluster cuyo centroide este mas cerca por distancia euclidea
        labels = [i for i in range(self.k)]
        self.space_dim = train_data.shape[1]
        self.centroids = []
        classification = []

        for entry in train_data:
            # we assign labels randomly to each entry of the classificator
            label = random.choice(list(labels))
            classification.append((entry, label))

        iteration = 0
        prev_classification = None
        # while iteration < self.max_iter:
        # The algorithm has converged when the assignments no longer change
        # The algorithm is often presented as assigning objects to the nearest cluster by distance
        while not self._check_tol(prev_classification, classification) and iteration < self.max_iter:

            prev_classification = classification.copy()
            classification = []

            # 2.
            # calcular centroide
            # asignar centroide al cluster cuyo centroide este mas cerca por distancia euclidea

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

    # los centroides son calculados como el promedio de las observaciones asignadas a cada cluster.
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
        # norm is l2 = The L2 norm that is calculated as the square root of the sum of the squared vector values.
        # euclidean
        distances = []
        for label, point in centroids:
            distance = np.linalg.norm(entry - point)
            distances.append((distance, label))
        # buscamos minimizarlo, tomamos la mas pequeña
        min_tuple = min(distances, key = lambda t: t[0])
        return min_tuple[1]

    # Cuando de un paso a otro no hay modificaciones, significa que se esta en presencia de un mınimo local.
    def _check_tol(self, prev_classification, new_classification):
        prev = 0
        new = 0

        if prev_classification == None or new_classification == None:
            return False

        for i in range(len(prev_classification)):
            prev += prev_classification[i][1]
        for i in range(len(new_classification)):
            new += new_classification[i][1]
        result = (new - prev)
        if result == 0:
            return True

        return False

    # def converge(self, prev_classification, new_classification):
    #     #check classifications
    #     print("coverage")
    #
    #     if prev_classification == None or new_classification == None:
    #         return False
    #
    #     for idx in range(len(prev_classification)):
    #         entry, prev_label = prev_classification[idx]
    #         entry, new_label = new_classification[idx]
    #         if prev_label != new_label:
    #             return False
    #
    #     return True




