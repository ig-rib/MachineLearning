import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from utils import regularGain, entropy, specificEntropy, findMostFrequentObjectiveValue, giniGain, classifyRow
from treeNode import Node
from copy import copy
import random as rd

percentage = .5

data = pd.read_csv('./TP2-ClasificacionSupervisada/titanic.csv', sep='\t')
obj = 'Survived'
attributeNames = ['Pclass', 'Sex', 'Age']
data = data[['Survived', 'Pclass', 'Sex', 'Age']]

data['Age'] = data['Age'].map(lambda x: x//15)
data['Age'] = data['Age'].fillna(data['Age'].mode()[0])

## a) Dividir en conjunto de entrenamiento y de prueba

# se mezcla el conjunto
data = data.sample(frac=1).reset_index(drop=True)

splittingIndex = int(len(data)*percentage)
training = data[:splittingIndex]
testSet = data[splittingIndex:]

## b) 
## Algoritmo ID3
## 1. Calcular la ganancia para cada atributo


objectiveValues = list(data[obj].unique())

def generateTree(training, gain, objectiveValues, attributes = None):

    usedAttributes = []
    if attributes == None:
        remainingAttributes = copy(attributeNames)
    else:
        remainingAttributes = copy(attributes)
    root = Node(None)
    F = [root]
    E = [root]
    while len(F) > 0:
        curr = F.pop()
        
        auxData = training
        for filter in curr.filters:
            auxData = auxData[ auxData[filter[0]] == filter[1]]
        gainsPerAttribute = {}
        if len(remainingAttributes) > 0:
            for aName in remainingAttributes:
                gainsPerAttribute[aName] = gain(training, [], obj, aName)
            currAttName = max(gainsPerAttribute, key=lambda key: gainsPerAttribute[key])
            remainingAttributes = [att for att in remainingAttributes if att != currAttName]
            curr.value = currAttName
            attValues = list(data[currAttName].unique())
            # print(auxData)
            for v in attValues:
                child = Node(curr)
                child.filters = copy(curr.filters)
                child.filters.append([currAttName, v])
                childAuxData = auxData[auxData[currAttName] == v]
                # En este caso el subset para este nodo tiene ejemplos,
                # es decir, hay casos que cumplen con este filtro
                # más específico
                if len(childAuxData) != 0:
                    ## Acá me fijo si todos los elementos del conjunto 
                    ## que me queda agregándole el filtro de este nodo
                    ## son del mismo valor del objetivo -> el nodo es hoja
                    leaf = False
                    for ov in objectiveValues:
                        if len(childAuxData[childAuxData[obj] == ov]) == len(childAuxData):
                            leaf = True
                            break
                    if leaf:
                        child.value = ov
                    else:
                        F.append(child)
                # En este caso, no hay nodos que cumplan con este nuevo filtro:
                # se crea una hoja cuyo valor es el más frecuente del atributo
                # objetivo en el conjunto del padre
                else:
                    child.value = findMostFrequentObjectiveValue(auxData, obj, objectiveValues)
                
                curr.children[v] = child
                E.append(child)
        # Si no me quedan atributos, los nodos que quedan son hojas
        else:
            curr.value = findMostFrequentObjectiveValue(auxData, obj, objectiveValues)
    return root

shannonTree = generateTree(training, regularGain, objectiveValues)
giniTree = generateTree(training, giniGain, objectiveValues)
print()

## Random Forest
## Se toman muestras del conjunto de training
# que tengan el mismo tamaño que éste.

denominator = 5
sampleSize = len(training) // denominator

trees = []
for i in range(denominator):
    currSample = training.sample(len(training), replace=True)
    # print(currSample)
    r = rd.randint(1, len(attributeNames))
    print('r', r)
    rd.shuffle(attributeNames)
    trees.append(generateTree(currSample, regularGain, objectiveValues, attributeNames[:r]))

trees

shannonCorrect = 0
giniCorrect = 0

for i in range(len(testSet)):
    row = testSet.iloc[i]

    print(row)
    shannonCorrect += 1 - np.abs(classifyRow(row, shannonTree) - row['Survived'])
    giniCorrect += 1 - np.abs(classifyRow(row, giniTree) - row['Survived'])

print('Shannon', shannonCorrect/len(testSet))
print('Gini', giniCorrect/len(testSet))
    
