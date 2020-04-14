#!/bin/python3

import pandas as pd

data = pd.read_excel('TP3-SVM/data/acath.xls')
data

# a)

percentage = 0.5
splittingIndex = int(percentage * len(data))
training = data[:splittingIndex]
testSet = data[splittingIndex:]

# b)

