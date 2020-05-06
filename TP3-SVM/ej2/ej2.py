#!/bin/python3

import pandas as pd
from sklearn import svm
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from utils import testSvm, plotConfusionMatrix

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

TPs, TNs, _,_,accuracy = testSvm(suppVectorMachine, testVectors, testYs)

plotConfusionMatrix(testYs, TPs, TNs, 'Confusion Matrix for SVM', 1)

# c)

svmHashPrecisions = {}

for C in [1, 10, 100, 1000]:
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        suppVectorMachine = svm.SVC(C = C, kernel = kernel)
        suppVectorMachine.fit(trainingVectors, trainingYs)
        TPs, TNs, totalClassifiedAsPos, totalClassifiedAsNeg, accuracy = testSvm(suppVectorMachine, testVectors, testYs)
        print(kernel, C, accuracy)
        svmHashPrecisions[(C, kernel)] = accuracy

print(max(svmHashPrecisions.keys(), key=lambda x: svmHashPrecisions[x]))