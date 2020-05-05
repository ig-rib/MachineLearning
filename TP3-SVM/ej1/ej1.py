#!/bin/python3

import random as rd
import numpy as np
import matplotlib.pyplot as plt

points = []
for i in range(100):
    points.append((rd.random()*5, rd.random()*5))

def f(x):
    return 1.5 * x + 0.25

def mapToClass(pair):
    if pair[1] > f(pair[0]): return 1
    else: return -1

r = 0.5
D = [ [p, mapToClass(p)] for p in points]

from linearPerceptron import SimpleStepPerceptron

perceptron = SimpleStepPerceptron(2, 0.5, -.25)
perceptron.train(D)

points = []
for i in range(1000):
    points.append((rd.random()*5, rd.random()*5))

D2 = [ [p, mapToClass(p)] for p in points]

c0 = 0
for x in D:
    pred = perceptron.classify(x[0])
    # print(f'Predicted vs Actual: {pred}  --- {x[1]} <-- {x}')
    c0 += 1 if x[1] == pred else 0
correct = 0
for x in D2:
    pred = perceptron.classify(x[0])
    # print(f'Predicted vs Actual: {pred}  --- {x[1]} <-- {x}')
    correct += 1 if x[1] == pred else 0
print(c0/len(D))
print(correct/len(D2))

x = np.linspace(0, 5, 100)
slope = -perceptron.w[0]/perceptron.w[1]
intercept = -perceptron.w0/perceptron.w[1]
y = [ xi*slope + intercept for xi in x ]
plt.plot(x, y)
red = [x[0] for x in D if x[1] == -1 and perceptron.classify(x[0]) == x[1]]
blue = [x[0] for x in D if x[1] == 1 and perceptron.classify(x[0]) == x[1]]
green = [x[0] for x in D if x[1] == -1 and perceptron.classify(x[0]) != x[1]]
orange = [x[0] for x in D if x[1] == 1 and perceptron.classify(x[0]) != x[1]]
plt.scatter([r[0] for r in red], [r[1] for r in red], color='red')
plt.scatter([b[0] for b in blue], [b[1] for b in blue], color='blue')
plt.scatter([b[0] for b in green], [b[1] for b in green], color='green')
plt.scatter([b[0] for b in orange], [b[1] for b in orange], color='orange')

plt.show()