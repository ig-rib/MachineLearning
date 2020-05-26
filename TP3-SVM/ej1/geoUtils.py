#!/bin/python3

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from itertools import combinations

## Gets N closest points to hyperplane for
## each class, when possible
## points: Array of [(x, y), Class]
## w: weights vector, including bias

def getNClosest(points, w, N=3):
    ## Inefficient but clearer
    classAPoints = [ p for p in points if p[1] == -1 ]
    classBPoints = [ p for p in points if p[1] == 1 ]
    b0 = w[0]
    b = w[1:]
    closestAPoints = sorted(classAPoints, key=lambda x: np.abs(np.dot(b, x[0]) - b0))[:N]
    closestBPoints = sorted(classBPoints, key=lambda x: np.abs(np.dot(b, x[0]) - b0))[:N]
    return closestAPoints, closestBPoints

def getBestHyperplane(SVCandidatesA, SVCandidatesB, D):
    
    hyps = auxGetHyperPlanes(SVCandidatesA, SVCandidatesB)
    hyps.extend(auxGetHyperPlanes(SVCandidatesB, SVCandidatesA))
    allPts = [ x for x in SVCandidatesA ]
    allPts.extend([ x for x in SVCandidatesB ])
    
    ash = [[(np.sign(h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']) != x[1]) for x in D] for h in hyps]
    ash1 = [list(filter(lambda x: x, [(np.sign(h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']) != x[1]) for x in D])) for h in hyps]
    ash2 = list(filter(lambda h: len(list(filter(lambda x: x, [(np.sign(h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']) != x[1]) for x in D]))) == 0, hyps))
    correctHyps = list(filter(lambda h: len(list(filter(lambda x: x, [np.sign(h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']) != np.sign(x[1]) for x in D]))) == 0, hyps))
    distances = [[ (h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']) * x[1] for x in D ] for h in hyps]

    return max(hyps, key=lambda h: min( [ (h['m'] * x[0][0]/ h['norm'] - x[0][1]/ h['norm']  + h['b']/h['norm']) * x[1] for x in D ] )), hyps

def auxGetHyperPlanes(setA, setB):
    hyperplanes = []

    for pair in combinations(setA, 2):
        for bCand in setB:
            SVA = pair
            SVB = bCand

            y2 = SVA[1][0][1]
            y1 = SVA[0][0][1]
            x2 = SVA[1][0][0]
            x1 = SVA[0][0][0]

            m = (y2 - y1) / (x2 - x1)
            bA = -x1 * m + y1
            bB = SVB[0][1] - SVB[0][0] * m

            bEnd = min(bA, bB) + np.abs(bA-bB) / 2
            hyperplanes.append({'m' : m, 'b' : bEnd, 'norm' : np.linalg.norm([m, -1])})

    return hyperplanes