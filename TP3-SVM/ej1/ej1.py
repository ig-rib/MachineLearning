#!/bin/python3

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import sys
import geoUtils as gU

points = []
for i in range(30):
    points.append((rd.random()*5, rd.random()*5))

def f(x):
    return 1.5 * x + 0.25

def mapToClass(pair):
    if pair[1] > f(pair[0]): return -1
    else: return 1

r = 0.5
D = [ [p, mapToClass(p)] for p in points]

from linearPerceptron import SimpleStepPerceptron

def classifyAndTest(D, epochs=1000):
    perceptron = SimpleStepPerceptron(len(D[0][0]), 0.005, .25)
    perceptron.train(D, minError=0.0, epochs=epochs)

    points = []
    for i in range(100):
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
    print('Accuracy with training set:', c0/len(D))
    print('Accuracy with test set:', correct/len(D2))

    x = np.linspace(0, 5, 100)
    slope = -perceptron.w[1]/perceptron.w[2]
    intercept = perceptron.w[0]/perceptron.w[2]
    print(f'\nSeparating Line Equation:\n{slope}*x + {intercept}\n\n')
    y = [ xi*slope + intercept for xi in x ]
    plt.plot(x, y)
    red = [x[0] for x in D2 if x[1] == -1 and perceptron.classify(x[0]) == x[1]]
    blue = [x[0] for x in D2 if x[1] == 1 and perceptron.classify(x[0]) == x[1]]
    green = [x[0] for x in D2 if x[1] == -1 and perceptron.classify(x[0]) != x[1]]
    orange = [x[0] for x in D2 if x[1] == 1 and perceptron.classify(x[0]) != x[1]]
    plt.scatter([r[0] for r in red], [r[1] for r in red], color='red')
    plt.scatter([b[0] for b in blue], [b[1] for b in blue], color='blue')
    plt.scatter([b[0] for b in green], [b[1] for b in green], color='green')
    plt.scatter([b[0] for b in orange], [b[1] for b in orange], color='orange')
    plt.title('Test Set Results')
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    plt.show()
    return perceptron

print('\n######################################################\nCORRECTLY CLASSIFIED TRAINING SET\n######################################################\n')
perceptron = classifyAndTest(D)

clA, clB = gU.getNClosest(D, perceptron.w, 3)



bestHyp, hyps = gU.getBestHyperplane(clA, clB)
for hyp in [bestHyp]:
    red = [x[0] for x in D if x[1] == -1 and perceptron.classify(x[0]) == x[1]]
    blue = [x[0] for x in D if x[1] == 1 and perceptron.classify(x[0]) == x[1]]
    plt.scatter([r[0] for r in red], [r[1] for r in red], color='red')
    plt.scatter([b[0] for b in blue], [b[1] for b in blue], color='blue')
    plt.scatter([r[0][0] for r in clA], [r[0][1] for r in clA], color='green')
    plt.scatter([b[0][0] for b in clB], [b[0][1] for b in clB], color='orange')
    x = np.linspace(0, 5, 100)
    slope = -perceptron.w[1]/perceptron.w[2]
    intercept = perceptron.w[0]/perceptron.w[2]
    print(f'\nSeparating Line Equation:\n{slope}*x + {intercept}\n\n')
    y = [ xi*slope + intercept for xi in x ]
    plt.plot(x, y)
    y = [ xi*hyp['m'] + hyp['b'] for xi in x ]
    plt.plot(x, y)
    print(f'\nSeparating Line Equation:\n{hyp["m"]}*x + {hyp["b"]}\n\n')
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    plt.show()
# points = []
# for i in range(1000):
#     points.append((rd.random()*5, rd.random()*5))
# D = [ [p, mapToClass(p)] for p in points]
# points = []
# for i in range(100):
#     x = rd.random()*5
#     points.append((x, f(x) - rd.random()*0.1 * (-1 if rd.random() < 0.5 else 1)))
# D.extend([[p, mapToWrongClass(p)] for p in points])

# print('\n######################################################\nTRAINING SET WITH SOME INCORRECTLY CLASSIFIED EXAMPLES\n######################################################\n')
# classifyAndTest(D, epochs=250)
