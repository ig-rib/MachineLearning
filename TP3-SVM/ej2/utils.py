#!/bin/python3

import pandas as pd
import matplotlib.image as img
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

def loadImage(filename):
    return pd.DataFrame(img.imread('data/cielo.jpg').reshape(-1, 3))

def normalize(dataFrame):
    minMaxScaler = MinMaxScaler()
    return pd.DataFrame(minMaxScaler.fit_transform(dataFrame), index=dataFrame.index, columns=['R', 'G', 'B'])


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
    accuracy = correct/len(testVectors)
    return TPs, TNs, totalClassifiedAsPos, totalClassifiedAsNeg, accuracy

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