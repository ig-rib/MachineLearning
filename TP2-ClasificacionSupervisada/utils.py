import math
import pandas as pd
import numpy as np
from scipy import stats

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

## No estaba en las diapos pero la ganancia de Gini
## es suma (para cada valor v del atributo A) de 
## (|Sv|/|S|) * giniCoeff(Sv)

def giniGain(dataFrame, filters, objective, attribute):
    auxData = dataFrame
    for filter in filters:
        auxData = auxData[ auxData[filter[0]] == filter[1]]
    sum = 0.0
    attributeValues = list(dataFrame[attribute].unique())
    for v in attributeValues:
        sum += len(dataFrame[dataFrame[attribute] == v]) * giniIndex(dataFrame[dataFrame[attribute] == v], objective)
    sum /= len(dataFrame)
    return sum
def giniIndex(dataFrame, objective):
    objectiveValues = list(dataFrame[objective].unique())
    ps = calculateProbabilities(dataFrame, objective, objectiveValues)
    sum = 1.0
    for ov in objectiveValues:
        sum -= ps[ov] ** 2
    return sum

def entropy(dataFrame, objective):
    objectiveValues = list(dataFrame[objective].unique())
    ps = calculateProbabilities(dataFrame, objective, objectiveValues)
    ## cálculo de la entropía
    sum = 0.0
    for ov in objectiveValues:
        if ps[ov] != 0:
            sum -= ps[ov] * math.log2(ps[ov])
    return sum

def specificEntropy(dataFrame, objective, attribute, attributeValue):
    objectiveValues = list(dataFrame[objective].unique())
    ## repite codigo pero ilustra cómo se calculan las probabilidades
    ps = calculateProbabilities(dataFrame, objective, objectiveValues)
    sum = 0.0
    for ov in objectiveValues:
        ## P(A = v | Obj = j) = P(A=v, Obj=j) / P(Obj=j)
        pjv = len(dataFrame[ (dataFrame[attribute] == attributeValue) & (dataFrame[objective] == ov )]) / len(dataFrame[dataFrame[objective] == ov])
        if pjv != 0:
            sum -= pjv * math.log2(pjv)
    return sum

def findMostFrequentObjectiveValue(dataFrame, objective, objectiveValues):
    ps = calculateProbabilities(dataFrame, objective, objectiveValues)
    return max(ps, key=lambda key: ps[key])


def calculateProbabilities (dataFrame, objective, objectiveValues):
    ps = {}
    for ov in objectiveValues:
        ps[ov] = len(dataFrame[dataFrame[objective] == ov])/len(dataFrame)
    return ps



def classifyRow(dataFrameRow, decisionTree):
    curr = decisionTree
    done = False
    while not done:
        if len(curr.children) > 0:
            curr = curr.children[dataFrameRow[curr.value]]
        else:
            done = True
    return curr.value

def randomForestClassifyRow(dataFrameRow, randomForest):
    results = [ classifyRow(dataFrameRow, decisionTree) for decisionTree in randomForest ]
    return stats.mode(results)[0]