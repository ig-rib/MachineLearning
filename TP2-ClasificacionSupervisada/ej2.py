import pandas as pd
import math
from math import sqrt
import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt

data = pd.read_csv('reviews_sentiment.csv', sep=';')

data = data.sample(frac=1).reset_index(drop=True)[1:5000]

# replace null values for 0
data.fillna(0.5, inplace = True) 
# replace negative and positive values for 1 and 2 
data.replace(['negative', 'positive'], [0, 1], inplace=True)


# increasing percetage, increases training 
percentage = 0.6
splittingIndex = int(len(data)*percentage)
training = data[:splittingIndex]
testSet = data[splittingIndex:]

# print(len(training.iloc[0][:]))


word_count = 0
count = 0

for i in range(len(data)):
	star_rating = data.iloc[i]["Star Rating"]
	if(star_rating==1):
		word_count += data.iloc[i]["wordcount"]
		count+=1
average = word_count/count
print("a) Los reviews valorados con 1 estrella tienen en promedio " + str(average) + " palabras")

# Euclidean distance between two vectors
# wordcount, title_sentiment, text_sentiment
def euclidean_distance (row1, row2):
	distance = 0.0
	for i in range(len(training.columns)):
		if i in (2, 3, 4, 6):
			distance+=(row1[i]-row2[i])**2
	return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for i in range(len(train)):
		train_row = train.iloc[i][:]
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def get_neighbors_weighted(train, test_row, num_neighbors):
	distances = list()
	for i in range(len(train)):
		train_row = train.iloc[i][:]
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][:])
	return neighbors

# Forma 1 
# def predict_classification(train, test_row, num_neighbors):
# 	neighbors = get_neighbors(train, test_row, num_neighbors)
# 	output_values = [row[-2] for row in neighbors]
# 	prediction = max(output_values, key=output_values.count)
# 	return prediction

# Forma 2 
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-2] for row in neighbors]
	star_prediction = [0, 0 ,0 ,0 ,0]

	for i in range(len(neighbors)):
		star = neighbors[i][5] 
		star_prediction[star-1]+=1

	max_rating = max(star_prediction)
	print("star_prediction index: " + str(star_prediction))
	for i in range(len(star_prediction)):
		if(star_prediction[i] == max_rating):
			prediction = i + 1
	return prediction

def predict_weighted_classification(train, test_row, num_neighbors, print_value):
	neighbors = get_neighbors_weighted(train, test_row, num_neighbors)
	w_neighbors = list()
	prediction = 0
	star_prediction = [0, 0 ,0 ,0 ,0]

	# asocio cada neighbor con su 1/d(xi, xq)^2
	for i in range(len(neighbors)):
		if(neighbors[i][1] == 0):
			w_neighbors.append(1)
		else:
			w_neighbors.append(1/(neighbors[i][1]**2))

	for i in range(len(neighbors)):
		star = (neighbors[i][0])[5] 
		star_prediction[star-1]+=1*(w_neighbors[i])

	max_rating = max(star_prediction)
	if print_value == 1: 
		print("star_prediction index: " + str(star_prediction))
	for i in range(len(star_prediction)):
		if(star_prediction[i] == max_rating):
			prediction = i + 1
	return prediction


# KNN
# 1. Tomo el conjunto de training 
# 2. Tomo una row de testeo 
# 3. Calculo las distancias
# 4. Tomo los 5 vecinos mas cercanos 
# 5. Predigo, con max sobre rating 

num_neighbors = 5

# puede ser un numero random tomo el 5 porque si, magic number
vec_analizar = testSet.iloc[5][:]
label = predict_classification(training, vec_analizar, num_neighbors)
print("KNN")
print('Data=%s, Predicted: %s' % (vec_analizar, label))
print('--------------------------------------------------------')


# KNN con distancia ponderada
# 1. Tomo el conjunto de trainin
# 2. Tomo una row de testo
# 3. Calculo las distancias
# 4. Tomos los 5 vecinos mas cercanos
# 5. Predigo, teniendo en cuenta la distancia, para obtener el max rating


label = predict_weighted_classification(training, vec_analizar, num_neighbors, 1)
print("Weighted KNN")
print('Data=%s, Predicted: %s' % (vec_analizar, label))
print('--------------------------------------------------------')


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
    confusionMatrix[roww['Star Rating']][predict_weighted_classification(training, roww, 5, 0)] += 1

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




