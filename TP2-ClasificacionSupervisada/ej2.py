import pandas as pd
from math import sqrt
from random import shuffle

data = pd.read_csv('reviews_sentiment.csv', sep=';')

data = data.sample(frac=1).reset_index(drop=True)[1:5000]

# replace null values for 0
data.fillna(0.5, inplace = True) 
# replace negative and positive values for 1 and 2 
data.replace(['negative', 'positive'], [0, 1], inplace=True)


# increasing percetage, increases training 
percentage = 0.4
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

def predict_weighted_classification(train, test_row, num_neighbors):
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


label = predict_weighted_classification(training, vec_analizar, num_neighbors)
print("Weighted KNN")
print('Data=%s, Predicted: %s' % (vec_analizar, label))
print('--------------------------------------------------------')





