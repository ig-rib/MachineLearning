#!/bin/python3

import random as rd

points = []
for i in range(1000):
    points.append((rd.random()*5, rd.random()*5))

def f(x):
    return 1.5 * x + 0.25

def mapToClass(pair):
    if pair[1] > f(pair[0]): return 1
    else: return -1

r = 0.5
D = [ [p, mapToClass(p)] for p in points]

from linearPerceptron import SimplePerceptron

perceptron = SimplePerceptron(2, 0.5, 0)
perceptron.train(D)

print(perceptron.classify((1, 4)))
print(perceptron.classify((1, 2)))
print(perceptron.classify((0.5, 0.905)))
print(perceptron.classify((1, 1)))