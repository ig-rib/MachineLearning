#!/bin/python3

import os
import matplotlib.image as img
import matplotlib.pyplot as plt
import utils as u
import pandas as pd
from sklearn.svm import SVC
import numpy as np

# a) Construir el conjunto de datos

percentage = 0.5

skyData = u.normalize(u.loadImage('data/cielo.jpg'))
cowData = u.normalize(u.loadImage('data/vaca.jpg'))
grassData = u.normalize(u.loadImage('data/pasto.jpg'))
testImageData = u.normalize(u.loadExactImage('data/cow.jpg', percentage = 1))
otherImageData = u.normalize(u.loadExactImage('data/otherImage.jpg', percentage = 1))

skyData["Class"] = "Sky"
cowData["Class"] = "Cow"
grassData["Class"] = "Grass"

theData = pd.concat([skyData, cowData, grassData], ignore_index=True)

# b) Dividir en training y test set

splittingIndex = int(len(theData) * percentage)
theData = theData.sample(frac=1, replace=False)
trainingSet = theData[:splittingIndex]
trainingX = trainingSet[['R', 'G', 'B']]
trainingY = trainingSet['Class']
test = theData[splittingIndex:]
testX = test[['R', 'G', 'B']]
testY = test['Class']

# c) Evaluar varias configuraciones de SVMs, matrices de confusión

svmHashPrecisions = {}

for C in [1]:
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        suppVectorMachine = SVC(C = C, kernel = kernel)
        suppVectorMachine.fit(trainingX, trainingY)
        accuracy, confusionMatrix = u.testSvm(suppVectorMachine, testX, testY)
        print(confusionMatrix)
        u.showConfusionMatrix(confusionMatrix, confusionMatrix.keys(), f'Confusion Matrix for {kernel}, C={C}')
        svmHashPrecisions[(C, kernel)] = accuracy
        print(C, kernel, accuracy)

# d) Mejor núcleo (y C)

bestSettings = max(svmHashPrecisions.keys(), key=lambda x: svmHashPrecisions[x])
print('Best Settings', bestSettings)

# e) Con el mejor clasificar todos los pixeles de la imagen

bestSVM = SVC(C = bestSettings[0], kernel = bestSettings[1])

classIdentifiers = {
    'Cow': [255, 0, 0],
    'Sky': [0, 0, 255],
    'Grass': [0, 255, 0]
}

bestSVM.fit(trainingX, trainingY)

def plotImage(imageData, w, h):
    imageMapping = bestSVM.predict(imageData)
    imageMapping.reshape(h, w)
    pixels = np.array([ classIdentifiers[x] for x in imageMapping ]).reshape(h, w, 3)
    plt.imshow(pixels)
    plt.show()

# plotImage(testImageData, 1140, 760)

# f) Con el mejor clasificar todos los pixeles de la otra imagen

plotImage(otherImageData, 1200, 800)