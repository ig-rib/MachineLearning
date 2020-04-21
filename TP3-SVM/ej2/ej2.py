#!/bin/python3

import pandas as pd
from sklearn import svm
import random as rd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('TP3-SVM/data/acath.xls')
data = data.sample(frac=1)
choleste = 'choleste'
# data[choleste] = data[choleste].fillna(data[choleste].mean())
data = data.apply(lambda x: x.fillna(x.mean()),axis=0)
# a)

percentage = 0.75
splittingIndex = int(percentage * len(data))
training = data[:splittingIndex]
testSet = data[splittingIndex:]


# b)

objective = 'sigdz'

trainingVectors = training.loc[:, training.columns != objective]
trainingYs = training[objective]

## possible kernel values are 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

suppVectorMachine = svm.SVC(C = 1.0, kernel = 'linear')
suppVectorMachine.fit(trainingVectors, trainingYs)

testVectors = testSet.loc[:, testSet.columns != objective]
testYs = testSet[objective]

predicted = suppVectorMachine.predict(testVectors)
totalClassifiedAsPos = 0
totalClassifiedAsNeg = 0
TPs = 0
TNs = 0
for i in range(len(testVectors)):
    predictedValue = suppVectorMachine.predict(np.array(testVectors.iloc[i]).reshape(1, -1))[0]
    actualValue = testYs.iloc[i]
    if predictedValue == 1: 
        totalClassifiedAsPos += 1
        if predictedValue == actualValue:
            TPs += 1
    elif predictedValue == 0: 
        totalClassifiedAsNeg += 1
        if predictedValue == actualValue:
            TNs += 1

def plotConfusionMatrix(testSet, TPs, TNs, title, figNo):
    matrix = []
    matrix.append([TPs, len(testSet[testSet == 1]) - TPs])
    matrix.append([len(testSet[testSet == 0]) - TNs, TNs])
    print(matrix)
    fig, ax = plt.subplots()
    ax.matshow(matrix)
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.1'))
    ax.set_xticklabels(['', 'Positives', 'Negatives'])
    ax.set_yticklabels(['', 'Positives', 'Negatives'])
    plt.title(title)
    plt.show()

plotConfusionMatrix(testYs, TPs, TNs, 'Confusion Matrix for SVM', 1)

# c)