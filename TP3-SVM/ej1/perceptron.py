import numpy as np
import sys

class Perceptron:

    def __init__(self, dimension, r, w0 = 0.1):
        self.w = np.array(dimension * [0])
        self.w0 = w0
        self.r = r

    def train(self, trainingSet, r=0):
        print('Train unimplemented')

    def classify(self, vector):
        print('Classify unimplemented')


class SimpleNonLinearPerceptron(Perceptron):

    def __init__(self, dimension, r, g, gPrime, w0 = 0.1):
        super().__init__(dimension, r, w0)
        self.g = g
        self.gPrime = gPrime
        self.w = np.random.random(dimension+1)
    
    def train(self, X, r = None, minError = 5e-4, epochs = sys.maxsize):
        if r != None:
            self.r = r
        error = sys.maxsize
        bestError = error
        i = 0
        actualYs = [ x[1] for x in X ]
        points = [ [[-1, *x[0]], x[1]] for x in X ]
        while error > minError and i < epochs:
            for x in points:
                excitation = np.dot(self.w, x[0])
                prime = self.gPrime(excitation)
                deltaW = (x[1] - self.g(excitation)) * prime * self.r * np.array(x[0])
                self.w = self.w + deltaW
            currentClassifications = [self.classify(y[0]) for y in points]
            squerr = 0
            for j in range(len(points)):
                currClass = self.classify(points[j][0])
                realClass = points[j][1]
                squerr = squerr + (realClass - currClass) ** 2
            # absSubtractions = [ np.abs(err) ** 2 for err in np.subtract(actualYs, currentClassifications) ]
            # error = sum(absSubtractions) / 2
            error = squerr / 2
            if error < bestError:
                bestError = error
                self.bestW = self.w
            # print(self.bestW, self.w, error)
            i += 1
        self.w = self.bestW

    def classify(self, x):
        if len(x) < len(self.w):
            y = [-1, *x]
            return self.classify(y)
        return self.g(np.dot(self.w, x))
        # return self.g(np.dot(self.w, x )+ self.w0)
