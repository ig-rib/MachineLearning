#!/bin/python3

import os
import pickle
import utils as u
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import image

# a) Construir el conjunto de datos

percentage = 0.5

# base_path = 'data'
base_path = '/Users/martinascomazzon/Documents/GitHub/MachineLearning/TP3-SVM/data'

def load_data(base_path):
    labels = []
    data = []
    for label in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, label)):
            for img in os.listdir(os.path.join(base_path, label)):
                img = image.imread(os.path.join(base_path, label, img))
                for row in img:
                    for col in row:
                        data.append(col)
                        labels.append(label)

    unique_lables = ['cow', 'grass', 'sky']
    categorical_lables = [i for i in unique_lables]
    labels = [categorical_lables.index(lable) for lable in labels]
    labels = np.array(labels)
    data = np.array(data)
    return data, labels

data, labels = load_data(base_path)

# b) Dividir en training y test set


trainingSet, test, trainingY, testY = train_test_split(data, labels, train_size=percentage)

# c) Evaluar varias configuraciones de SVMs, matrices de confusión

suppVectorMachine = SVC(C = 0.8, kernel = 'lineal')
suppVectorMachine.fit(trainingSet, trainingY)

predictions = suppVectorMachine.predict(test)

possible_labels = set(testY)
possible_labels = list(possible_labels)

#confusion matrix
conf_matrix = confusion_matrix(testY, predictions, labels=possible_labels)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix, index=list(possible_labels), columns=list(possible_labels))
sn.heatmap(df_cm, annot=True)

# d) Mejor núcleo (y C)
# mejor nucleo es el de rbf

# e) Con el mejor clasificar todos los pixeles de la imagen

class_color = [
    [255, 0,0],
    [0, 255, 0],
    [0, 0, 255]
]

# img_path = 'data/image_cow.jpg'
img_path = '/Users/martinascomazzon/Documents/GitHub/MachineLearning/TP3-SVM/data/image_cow.jpg.jpg'
img = image.imread(img_path)

predictions_img = []
for row in img:
    row_pred = []
    for col in row:
        pred = suppVectorMachine.predict(np.array([col]))
        row_pred.append(class_color[int(pred)])
        print(row_pred)
    predictions_img.append(row_pred)

predictions_img = np.array(predictions_img)
predictions_img = predictions_img.astype('uint8')
image.imsave('output.jpg', predictions_img)
