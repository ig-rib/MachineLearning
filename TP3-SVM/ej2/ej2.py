#!/bin/python3

import pandas as pd
from sklearn import svm
import random as rd

data = pd.read_excel('TP3-SVM/data/acath.xls')
data = data.sample(frac=1)
choleste = 'choleste'
# data[choleste] = data[choleste].fillna(data[choleste].mean())
data.apply(lambda x: x.fillna(x.mean()),axis=0)
# a)

percentage = 0.5
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
predicted