#!/bin/python3

import numpy as np
import sys

class SimplePerceptron:

    def __init__(self, dimension, r, biasLimit = 0):
        self.w = np.array(dimension * [0])
        self.b = dimension * [biasLimit]
        self.r = r
    
    def train(self, X, r = None, minError = 1e-3, epochs = 10):
        if r != None:
            self.r = r
        error = sys.maxsize
        bestError = error
        i = 0
        actualYs = [ x[1] for x in X ]
        points = [ x[0] for x in X ]
        while error > minError and i < epochs:
            for x in X:
                predictedY = np.dot(self.w, x[0])
                np.add(predictedY, self.b)
                deltaW = (x[1] - np.sign(predictedY)) * self.r * np.array(x[0])
                self.w = self.w + deltaW
            currentClassifications = [self.classify(y) for y in points]
            error = sum([ np.abs(err) for err in np.subtract(actualYs, currentClassifications) ]) / len(X)
            if error < bestError:
                bestError = error
                self.bestW = self.w
            print(self.bestW, self.w, error)
            i += 1

    def classify(self, x):
        return np.sign(np.dot(self.w, x))