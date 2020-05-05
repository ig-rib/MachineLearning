#!/bin/python3

import pandas as pd
from sklearn import svm
import random as rd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('data/acath.xls')
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

suppVectorMachine = svm.SVC(C = 1000, kernel = 'poly')
suppVectorMachine.fit(trainingVectors, trainingYs)

testVectors = testSet.loc[:, testSet.columns != objective]
testYs = testSet[objective]

# correct = 0
# PredYs = suppVectorMachine.predict(trainingVectors)
# for i in range(len(trainingYs)):
#     print(PredYs[i], trainingYs.iloc[i])
#     correct+= 1 if PredYs[i] == trainingYs.iloc[i] else 0
# print(correct/len(trainingVectors))

## TODO extract method

def testSvm(svm, testVectors, testYs):
    predYs = svm.predict(testVectors)
    totalClassifiedAsPos = 0
    totalClassifiedAsNeg = 0
    TPs = 0
    TNs = 0
    correct = 0
    for i in range(len(testVectors)):
        predictedValue = predYs[i]
        actualValue = testYs.iloc[i]
        if predictedValue == 1: 
            totalClassifiedAsPos += 1
            if predictedValue == actualValue:
                TPs += 1
        elif predictedValue == 0: 
            totalClassifiedAsNeg += 1
            if predictedValue == actualValue:
                TNs += 1
        correct+= 1 if predYs[i] == testYs.iloc[i] else 0
    print(correct/len(testVectors))
    return TPs, TNs, totalClassifiedAsPos, totalClassifiedAsNeg, correct

TPs, TNs, _,_,_ = testSvm(suppVectorMachine, testVectors, testYs)

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

svmHashPrecisions = {}

for C in [1, 10, 100, 1000]:
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        suppVectorMachine = svm.SVC(C = C, kernel = 'poly')
        suppVectorMachine.fit(trainingVectors, trainingYs)
        TPs, TNs, totalClassifiedAsPos, totalClassifiedAsNeg, correct = testSvm(suppVectorMachine, testVectors, testYs)
        svmHashPrecisions[(C, kernel)] = TPs / totalClassifiedAsPos

print(max(svmHashPrecisions.keys(), key=lambda x: svmHashPrecisions[x]))