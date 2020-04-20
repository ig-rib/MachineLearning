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

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-2] for row in neighbors]
	# print("output_values: " + str(output_values))
	prediction = max(output_values, key=output_values.count)
	return prediction

# 1. Tomo el conjunto de training 
# 2. Tomo una row de testeo 
# 3. Calculo las distancias
# 4. Tomo los 5 vecinos mas cercanos 
# 5. Predigo, con max sobre rating 

num_neighbors = 5
vec_analizar = testSet.iloc[5][:]
label = predict_classification(training, vec_analizar, num_neighbors)
print('Data=%s, Predicted: %s' % (vec_analizar, label))
print('--------------------------------------------------------')







