#!/bin/python3

import numpy as np
import sys
import random as rd

from perceptron import Perceptron

class SimpleStepPerceptron(Perceptron):

    def __init__(self, dimension, r, w0 = 0.1):
        super().__init__(dimension, r, w0)
        self.w = np.random.random(dimension+1)

    def train(self, X, r = None, minError = 1e-3, epochs = 1000):
        if r != None:
            self.r = r
        error = sys.maxsize
        bestError = error
        i = 0
        actualYs = [ x[1] for x in X ]
        points = [ [[-1, *x[0]], x[1]] for x in X ]
        while error > minError and i < epochs:
            for x in points:
                predictedY = np.dot(self.w, x[0])
                # predictedY += self.w0
                activation = np.sign(predictedY)
                deltaW = (x[1] - activation) * self.r * np.array(x[0])
                self.w = self.w + deltaW
            currentClassifications = [self.classify(y[0]) for y in points]
            absSubtractions = [ np.abs(err) for err in np.subtract(actualYs, currentClassifications) ]
            error = sum(absSubtractions) / 2
            if error < bestError:
                bestError = error
                self.bestW = self.w
            print(self.w)
            i += 1
        self.w = self.bestW

    def classify(self, x):
        if len(x) < len(self.w):
            y = [-1, *x]
            return self.classify(y)
        return 1 if np.sign(np.dot(self.w, x)) > 0 else -1 