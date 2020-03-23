#!/bin/python3

import os
import random
import pandas as pd
import numpy as np
import re

base = './Docu'
percentage = 0.10



# print()

# data = pd.read_excel('./Noticias_argentinas.xlsx')
data = pd.read_csv('./aa_bayes.tsv', sep='\t')
data = data[( data['categoria'] != 'Destacadas') & (data['categoria'] != 'Noticias destacadas') & (data['categoria'] != 'NaN')].dropna()


splittingIndex = int(len(data)*percentage)
training = data[:splittingIndex]
testSet = data[splittingIndex:]

# Preparar las estructuras de datos
wordCount = {}
totalWordCount = {}
categories = [x for x in data.categoria.unique() if x is not np.nan]
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

# Calcular frecuencias

for cat in wordCount.keys():
    for word in wordCount[cat]:        
        wordCount[cat][word] = (wordCount[cat][word] + 1) / (totalWordCount[cat] + len(categories))

# Matriz de Confusion
correct = 0
incorrect = 0
confusionMatrix = {}

for cat1 in sorted(categories):
    confusionMatrix[cat1] = {}
    for cat2 in sorted(categories):
        confusionMatrix[cat1][cat2] = 0

for i in range (len(testSet)):
    test = testSet.iloc[i]['titular'].split(" ")
    trueCat = testSet.iloc[i]['categoria']
    # if trueCat is not np.nan:
    categoryProbabilities = {}
    predictedCat = ''
    maxProb = 0
    for cat in categories:
        categoryProbabilities[cat] = 1
    for cat in categories:
        # P(titular | clase)
        for word in test:
            word = re.sub(r'[^(\w|\-)]', '', word).lower()
            if word != '' and len(word) > 3:
                #Laplace Smoothing
                categoryProbabilities[cat] *= (( wordCount[cat].get(word, 0) + 1 )/ (totalWordCount[cat] + len(categories)))
        # P(titular | clase) * P(titular)
        # categoryProbabilities[cat] *= len(wordCount[cat].keys()) / len(data)
        categoryProbabilities[cat] *= len(wordCount[cat].keys()) / 1000
        if categoryProbabilities[cat] > maxProb:
            maxProb = categoryProbabilities[cat]
            predictedCat = cat
    confusionMatrix[trueCat][predictedCat] += 1

truePositives = {}
totalClassifiedAs = {}
totalTrue = {}

for cat1 in categories:
    for cat2 in categories:
        if cat1 == cat2:
            truePositives[cat1] = truePositives.get(cat1, 0) + confusionMatrix[cat1][cat2]
        totalTrue[cat1] = totalTrue.get(cat1, 0) + confusionMatrix[cat1][cat2]
        totalClassifiedAs[cat2] = totalClassifiedAs.get(cat2, 0) + confusionMatrix[cat1][cat2]


for cat in categories:
    # Recall = TP / (TP + FN)
    recall = truePositives[cat]/totalTrue[cat]
    # Precision = TP / (TP + FP)
    precision = truePositives[cat]/totalClassifiedAs[cat]
    # falsePositiveRate = FP / ( FP + TN )
    falsePositiveRate = (totalClassifiedAs[cat]-truePositives[cat]) / (len(testSet)-totalTrue[cat])
    print('Category: %s\n\tTrue Positive Rate: %g\n\tFalse Positive Rate: %g\n\tRecall: %g\n\tPrecision: %g\n\tF1-Score: %g\n\n' % (cat, recall, falsePositiveRate, recall, precision, 2*precision*recall/(precision+recall)))

print('Correctly Classified: %d\nIncorrectly Classified: %d\n' % (correct, incorrect))

# 