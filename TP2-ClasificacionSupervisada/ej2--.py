#!/bin/python3
import pandas as pd
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from Levenshtein import distance as levenshtein_distance

data = pd.read_csv('reviews_sentiment.csv', sep=';')

allHeaders = ['Review Title', 'wordcount', 'titleSentiment', 'Star Rating', 'sentimentValue']
explVariables = ['Review Title', 'wordcount', 'titleSentiment', 'sentimentValue']
stringFields = ['Review Title', 'titleSentiment']

data = data[['Review Title', 'wordcount', 'titleSentiment', 'Star Rating', 'sentimentValue']]

percentage = 0.90
splittingIndex = int(len(data)*percentage)
data=data.sample(frac=1)
training = data[:splittingIndex]
testSet = data[splittingIndex:]

word_count = 0
count = 0

for i in range(len(data)):
	star_rating = data.iloc[i]["Star Rating"]
	if(star_rating==1):
		word_count += data.iloc[i]["wordcount"]
		count+=1
average = word_count/count
print("a) Los reviews valorados con 1 estrella tienen en promedio " + str(average) + " palabras")

## c)

def calculateDistance(row1, row2):
    sum = 0
    for var in explVariables:
        if var not in stringFields:
            sum += (row1[var] - row2[var]) ** 2
        elif var == 'Review Title':
            sum += (1-levenshtein_distance(row1[var], row2[var])/max(len(row1[var]), len(row2[var])))
        elif var == 'titleSentiment' and row1[var] == row2[var]:
                sum += 1
    return math.sqrt(sum)

def KNN(row, existing, K):
    distances = sorted([ (index, calculateDistance(row, existing.iloc[index])) for index in range(len(existing)) ], key=lambda x: x[1])[0:K]
    values = [ existing.iloc[elem[0]]['Star Rating'] for elem in distances ]
    return Counter(values).most_common(1)[0][0]

def WeightedKNN(row, existing, K):
    distances = sorted([ (index, calculateDistance(row, existing.iloc[index])) for index in range(len(existing)) ], key=lambda x: x[1])[0:K]
    zeroes = [ elem for elem in distances if elem[1] == 0 ]
    if len(zeroes) > 0:
        values = [ existing.iloc[elem[0]]['Star Rating'] for elem in distances ]
        return Counter(values).most_common(1)[0][0]
    else:
        values = [ (existing.iloc[elem[0]]['Star Rating'], elem[1], existing.iloc[elem[0]]['Star Rating']/elem[1]**2) for elem in distances ]
        totals = set([ (value[0], sum([val[2] for val in values if val[0] == value[0]])) for value in values])
        return max(totals, key=lambda x: x[1])[0]

## d)

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
        print('Category: %s\n\tTrue Positive Rate: %g\n\tFalse Positive Rate: %g\n\tFalse Negative Rate: %g\n\tRecall: %g\n\tPrecision: %g\n\tF1-Score: %g\n\n' 
        % (cat, recall, falsePositiveRate, falseNegativeRate, recall, precision, 2*precision*recall/(precision+recall)))


categories = sorted(list(data['Star Rating'].unique()))

correct = 0
incorrect = 0
confusionMatrix = {}
# Preparar la matriz de confusion
for cat1 in sorted(categories):
    confusionMatrix[cat1] = {}
    for cat2 in sorted(categories):
        confusionMatrix[cat1][cat2] = 0

for index, roww in testSet.iterrows():
    confusionMatrix[roww['Star Rating']][WeightedKNN(roww, training, 5)] += 1

def showConfusionMatrix(cM, cats):
    
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
    plt.title('Matriz de Confusion')
    plt.show()


showConfusionMatrix(confusionMatrix, categories)
printCategoryInfo(confusionMatrix, categories)

