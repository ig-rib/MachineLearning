#!/bin/python3

import matplotlib.pyplot as plt

def plotResults(trainingXys, testXys, title, figNo):
    plt.plot([p[0] for p in trainingXys], [p[1] for p in trainingXys], marker='o', linestyle='')
    plt.plot([p[0] for p in testXys], [p[1] for p in testXys], marker='o', linestyle='')
    plt.legend(['Training', 'Test'])
    plt.title(title)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Precision')
    plt.show()