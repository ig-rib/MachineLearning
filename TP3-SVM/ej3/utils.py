#!/bin/python3

import pandas as pd
import matplotlib.image as img
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

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

def showConfusionMatrix(cM, cats, title):
    
    plotMat = []

    for cat1 in cats:
        row = []
        for cat2 in cats:
            row.append(cM[cat1][cat2])
        plotMat.append(row)

    fig, ax = plt.subplots()
    ax.matshow(plotMat)

    for (i, j), z in np.ndenumerate(plotMat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.1'))

    plotCategories = ['']
    plotCategories.extend(cats)
    ax.set_xticklabels(plotCategories)
    ax.set_yticklabels(plotCategories)
    plt.title(title)
    plt.show()

    print('Pr/Ac', end='\t')
    for cat in cats:
        print(cat, end='\t')
    print()
    for cat1 in cats:
        print(cat1, end='\t')
        for cat2 in cats:
            print(cM[cat1][cat2], end='\t')
        print()
