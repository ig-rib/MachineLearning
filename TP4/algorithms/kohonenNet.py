#!/bin/python3

import numpy as np
import math
import random
from collections import namedtuple

def alpha(s, totalIterations):
    return 1 - s / totalIterations

def theta1(i, j, s, R=None):
    return np.abs(i - j) / (s+1)

def theta(i, j, s, L, R=2):
    winnerI = i // L
    winnerJ = i % L
    otherI = j // L
    otherJ = j % L
    return 1 if math.sqrt((winnerI-otherI) ** 2 + (winnerJ-otherJ) ** 2) <= R else 0
    # return 1 if np.abs(i-j) <= R else 0

# def theta2(i, j, D, R=2):

# Coord = namedtuple('Coord', ['x', 'y'])


## Implements a square Kohonen Net
## using euclidean distance between nodes

class KohonenNetwork:

    ## vectorDimension is the length of an input vector,
    ## which is the same as for weight vectors
    ## M is the number of nodes
    ## 
    def __init__(self, vectorDimension, M, D, randomWeights=False):
        self.vDim = vectorDimension
        self.M = M
        if randomWeights:
            self.W = np.asmatrix(np.random.random(size=(M, vectorDimension)))
        else:
            self.W = np.array([])
            idx = np.random.randint(D.shape[0], size = M)
            self.W = D[idx, :]

        self.L = math.sqrt(M) # The side of the square
    
    ## D is an np.array of samples
    ## lda (lambda) is the number of iterations
    ## R is the initial radius of neighbourhood
    ## Example: given neuron N1 at P1=(i1, j1) in the grid; neuron N2
    ## at P2=(i2, j2), they are only neighbours if norm2(P1-P2) <= R

    def train(self, D, lda, R=3):
        indices = list(range(D.shape[0]))
        it = 0
        activationCounts = {i : 0 for i in range(self.M)}
        while it < lda:
            random.shuffle(indices)
            for index in indices:
                curr = D[index]
                distances = [ [np.linalg.norm(curr - x), i] for i, x in enumerate(self.W) ]
                minIndex = min(distances, key=lambda x: x[0])[1]
                activationCounts[minIndex]+=1
                for rowNo in range(self.W.shape[0]):
                    if theta(minIndex, rowNo, it, self.L, R) > 0:
                        self.W[rowNo] = self.W[rowNo] + theta(minIndex, rowNo, it, self.L, R) * alpha(it, lda) * (curr - self.W[rowNo])
                if it % (lda*.1) == 0 and R > 1:
                    R -= 1
                it+=1
        return activationCounts
        
    def getClass(self, vector):
        distances = [ [np.linalg.norm(vector - x), i] for i, x in enumerate(self.W) ]
        neuronNo = min(distances, key=lambda x: x[0])[1]
        return (neuronNo // self.L, neuronNo % self.L)
        # return Coord(x=neuronNo // self.L, y=neuronNo % self.L)