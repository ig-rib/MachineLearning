#!/bin/python3

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

########################################################################
# FUNCIONES
########################################################################
def getConfusionMatrix(categories, testSet, training, totalWordCount, positiveCategory=None, threshold=1):
    # Matriz de Confusion
    correct = 0
    incorrect = 0
    confusionMatrix = {}
    # Preparar la matriz de confusion
    for cat1 in sorted(categories):
        confusionMatrix[cat1] = {}
        for cat2 in sorted(categories):
            confusionMatrix[cat1][cat2] = 0

    # Testear el clasificador de Bayes
    for i in range (len(testSet)):
        test = testSet.iloc[i]['titular'].split(" ")
        trueCat = testSet.iloc[i]['categoria']
        # if trueCat is not np.nan:
        categoryProbabilities = {}
        predictedCat = ''
        maxProb = 0
        for cat in categories:
            categoryProbabilities[cat] = 1
        # Calcular probabilidad a posteriori por cada clase
        for cat in categories:
            # P(titular | clase)
            for word in test:
                word = re.sub(r'[^(\w|\-)]', '', word).lower()
                if word != '' and len(word) > 3:
                    if cat == positiveCategory:
                        categoryProbabilities[cat] *= threshold
                    #Laplace Smoothing para casos nulos
                    categoryProbabilities[cat] *= ( wordCount[cat].get(word, 0) + 1 / (totalWordCount[cat] + len(categories) ) )
            # P(titular | clase) * P(clase)
            categoryProbabilities[cat] *= len(training[training['categoria'] == cat]) / len(training)
            # Checkear maxima probabilidad a posteriori
            if categoryProbabilities[cat] > maxProb:
                maxProb = categoryProbabilities[cat]
                predictedCat = cat
        confusionMatrix[trueCat][predictedCat] += 1
        if trueCat == predictedCat:
            correct += 1
        else:
            incorrect += 1
    return confusionMatrix, correct, incorrect

def printCategoryInfo(confusionMatrix, categories):

    truePositives = {}
    totalClassifiedAs = {}
    totalTrue = {}

    for cat1 in categories:
        truePositives[cat1] = confusionMatrix[cat1][cat1]
        for cat2 in categories:
            totalTrue[cat1] = totalTrue.get(cat1, 0) + confusionMatrix[cat1][cat2]
            totalClassifiedAs[cat2] = totalClassifiedAs.get(cat2, 0) + confusionMatrix[cat1][cat2]

    for cat in categories:
        # Recall = TP / (TP + FN)
        recall = truePositives[cat]/totalTrue[cat]
        falseNegativeRate = (totalTrue[cat] - truePositives[cat]) / totalTrue[cat]
        # Precision = TP / (TP + FP)
        precision = truePositives[cat]/totalClassifiedAs[cat]
        # falsePositiveRate = FP / ( FP + TN )
        falsePositiveRate = (totalClassifiedAs[cat]-truePositives[cat]) / (len(testSet)-totalTrue[cat])
        print('Category: %s\n\tTrue Positive Rate: %g\n\tFalse Positive Rate: %g\n\tFalse Negative Rate: %g\n\tRecall: %g\n\tPrecision: %g\n\tF1-Score: %g\n\n' % (cat, recall, falsePositiveRate, falseNegativeRate, recall, precision, 2*precision*recall/(precision+recall)))



########################################################################
# INICIALIZACION
########################################################################


base = './Docu'
percentage = 0.50

data = pd.read_excel('./Noticias_argentinas.xlsx')
# data = pd.read_csv('./aa_bayes.tsv', sep='\t')
data = data[( data['categoria'] != 'Destacadas') & (data['categoria'] != 'Noticias destacadas') & (data['categoria'] != 'NaN')].dropna()

data = data.sample(frac=1).reset_index(drop=True)[1:5000]

splittingIndex = int(len(data)*percentage)
training = data[:splittingIndex]
testSet = data[splittingIndex:]

########################################################################
# CLASIFICADOR DE BAYES
########################################################################

# Preparar las estructuras de datos
wordCount = {}
totalWordCount = {}
categories = sorted([x for x in data.categoria.unique() if x is not np.nan])
for cat in categories:
    wordCount[cat] = {}
    totalWordCount[cat] = 0

# Armar tabla de datos
# # for i in range(len(data)):
for i in range(len(training)):
    for word in training.iloc[i]['titular'].split(" "):
        # if training.iloc[i]['categoria'] is not np.nan:
            # for cat in categories:
            #     if wordCount[cat].get(word, None) == None:
            #         wordCount[cat][word] = 0
        word = re.sub(r'[^(\w|\-)]', '', word).lower()
        if word != '' and len(word) > 3:
            wordCount[training.iloc[i]['categoria']][word] = wordCount[training.iloc[i]['categoria']].get(word, 0) + 1
            totalWordCount[training.iloc[i]['categoria']] += 1

# Calcular frecuencias relativas
for cat in wordCount.keys():
    for word in wordCount[cat]: 
        # Laplace smoothing para casos no nulos       
        wordCount[cat][word] = (wordCount[cat][word] + 1) / (totalWordCount[cat] + len(categories))

########################################################################

########################################################################
# MATRIZ DE CONFUSION Y OTRAS METRICAS
########################################################################

confusionMatrix, correct, incorrect = getConfusionMatrix(categories, testSet, training, totalWordCount)
print(confusionMatrix)

plotMat = []

for cat1 in categories:
    row = []
    for cat2 in categories:
        row.append(confusionMatrix[cat1][cat2])
    plotMat.append(row)

fig, ax = plt.subplots()
ax.matshow(plotMat)

for (i, j), z in np.ndenumerate(plotMat):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.1'))

plotCategories = ['']
plotCategories.extend(categories)
ax.set_xticklabels(sorted(plotCategories))
ax.set_yticklabels(sorted(plotCategories))
plt.title('Matriz de Confusion')
plt.show()

printCategoryInfo(confusionMatrix, categories)
print('Correctly Classified: %d\nIncorrectly Classified: %d\n' % (correct, incorrect))

########################################################################
# CURVA ROC
########################################################################

positiveCat = categories[0]
ROCPoints = []
# evaluo con un conjunto del 10% del tamanio del testSet
newTestSet = testSet[1:len(testSet)//10]
for i in range(1,10):
    confusionMatrix, _, _ = getConfusionMatrix(categories, newTestSet, training, totalWordCount, positiveCat, i)
    truePositives = confusionMatrix[positiveCat][positiveCat]
    totalClassifiedAs = 0
    totalTrue = 0
    for cat in categories:
        totalClassifiedAs += confusionMatrix[cat][positiveCat]
        totalTrue += confusionMatrix[positiveCat][cat]
    TPR = truePositives / totalTrue
    FPR = (totalClassifiedAs-truePositives) / (len(newTestSet) - totalTrue)
    ROCPoints.append([FPR, TPR])

    # print(confusionMatrix)
plt.plot([x[0] for x in ROCPoints], [x[1] for x in ROCPoints])
auc = 0
for i in range(len(ROCPoints)-1):
    auc += max(ROCPoints[i+1][1], ROCPoints[i][1]) * np.abs(ROCPoints[i+1][0]-ROCPoints[i][0]) / 2
print('AUC: ', auc)
plt.title('Curva ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()