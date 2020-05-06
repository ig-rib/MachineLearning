#!/bin/python3

import pandas as pd
import matplotlib.image as img
from sklearn.preprocessing import MinMaxScaler

def loadImage(filename, percentage = 0.1):
    data = pd.DataFrame(img.imread(filename).reshape(-1, 3))
    return data.sample(frac=percentage, replace=False)

def loadExactImage(filename, percentage = 1):
    data = pd.DataFrame(img.imread(filename).reshape(-1, 3))
    return data[:int(len(data)*percentage)]

def normalize(dataFrame):
    minMaxScaler = MinMaxScaler()
    return pd.DataFrame(minMaxScaler.fit_transform(dataFrame), index=dataFrame.index, columns=['R', 'G', 'B'])


def testSvm(svm, testVectors, testYs):
    predYs = svm.predict(testVectors)

    correct = 0
    confusionMatrix = { header: { header1: 0 for header1 in ['Sky', 'Cow', 'Grass'] } for header in ['Sky', 'Cow', 'Grass'] }
    for i in range(len(testVectors)):
        predictedValue = predYs[i]
        actualValue = testYs.iloc[i]
        confusionMatrix[predYs[i]][actualValue] += 1
        correct+= 1 if predYs[i] == testYs.iloc[i] else 0
    accuracy = correct/len(testVectors)
    return accuracy, confusionMatrix

