import pandas as pd

data = pd.read_csv('reviews_sentiment.csv', sep=';')


percentage = 0.50
splittingIndex = int(len(data)*percentage)
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

