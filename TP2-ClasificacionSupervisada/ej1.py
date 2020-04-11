import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math


percentage = .5

data = pd.read_csv('./TP2-ClasificacionSupervisada/titanic.csv', sep='\t')
obj = 'Survived'
attributeNames = list(filter(lambda x: x in ['Pclass', 'Sex', 'Age'], list(data.keys())))
data = data[['Survived', 'Pclass', 'Sex', 'Age']]

data['Age'] = data['Age'].map(lambda x: x//10)
data['Age'] = data['Age'].fillna(data['Age'].mean())
## a) Dividir en conjunto de entrenamiento y de prueba


splittingIndex = int(len(data)*percentage)
training = data[:splittingIndex]
testSet = data[splittingIndex:]


## b) 


## Filters son pairs (attribute, attributeValue)
def regularGain(dataFrame, filters, objective, attribute):
    auxData = dataFrame
    for filter in filters:
        auxData = auxData[ auxData[filter[0]] == filter[1]]
    Hs = entropy(dataFrame, objective)
    sum = 0.0
    attributeValues = list(dataFrame[attribute].unique())
    for v in attributeValues:
        sum -= len(dataFrame[dataFrame[attribute] == v]) * specificEntropy(dataFrame, objective, attribute, v)
    sum /= len(dataFrame)
    return Hs + sum

def entropy(dataFrame, objective):
    objectiveValues = list(dataFrame[objective].unique())
    ps = {}
    ## repite codigo pero ilustra cómo se calculan las probabilidades
    for ov in objectiveValues:
        ps[ov] = len(dataFrame[dataFrame[objective] == ov])/len(dataFrame)
    ## cálculo de la entropía
    sum = 0.0
    for ov in objectiveValues:
        if ps[ov] != 0:
            sum -= ps[ov] * math.log2(ps[ov])
    return sum

def specificEntropy(dataFrame, objective, attribute, attributeValue):
    objectiveValues = list(dataFrame[objective].unique())
    ## repite codigo pero ilustra cómo se calculan las probabilidades
    ps = {}
    for ov in objectiveValues:
        ps[ov] = len(dataFrame[dataFrame[objective] == ov])/len(dataFrame)
    sum = 0.0
    for ov in objectiveValues:
        ## P(A = v | Obj = j) = P(A=v, Obj=j) / P(Obj=j)
        pjv = len(dataFrame[ (dataFrame[attribute] == attributeValue) & (dataFrame[objective] == ov )]) / len(dataFrame[dataFrame[objective] == ov])
        if pjv != 0:
            sum -= pjv * math.log2(pjv)
    return sum


## Algoritmo ID3
## 1. Calcular la ganancia para cada atributo

gainsPerAttribute = {}

for aName in attributeNames:
    gainsPerAttribute[aName] = regularGain(data, [], obj, aName)
gainsPerAttribute