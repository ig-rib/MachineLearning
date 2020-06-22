#!/bin/python3

import pandas as pd
from utils import plot_confusion_matrix_bis, fill_data, normalize_data
from algorithms.kmeans import *
from algorithms.kmeans2 import *
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# for confusion matrix printing
import seaborn as sns
import matplotlib.pyplot as plt
from algorithms.hierarchicalClustering import HierarchicalClustering, ClusterNode
from algorithms.kohonenNet import KohonenNetwork

# USEFULL LINKS
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



def logistic_training(ex, train_data, test_data, train_labels, test_labels):
    # trainig Logistic taken from sklearn.linear_model
    model = LogisticRegression(n_jobs=3, C=0.3);
    model.fit(train_data, train_labels)
    coefficients = model.coef_
    intercept = model.intercept_

    # testing results
    # a partir de las tres variables o cuatro, te dice si esta o no enfermo
    predictions = model.predict(test_data)
    title = 'Regresión Logística'
    if ex == 'b':
        title = title + ' sin sex (b)'
    if ex == 'd':
        title = title + ' con sex (d)'
    plot_confusion_matrix_bis(title, model, predictions, test_data, test_labels, normalize='None')
    return coefficients, intercept

# import data
file_data = pd.read_csv('data/acath.csv', sep=';')
file_data = fill_data(file_data)
data = file_data[['age', 'cad.dur', 'choleste']]
label = file_data['sigdz'].to_list()

# normalize data
data = normalize_data(data)
data = data.to_numpy()

# from sklearn.model_selection import train_test_split
# train_percentage = 0.8
# train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

# print('Ejercicio B, sin sex')
# coefficients, intercept = logistic_training('b', train_data, test_data, train_labels, test_labels)

# p_num_exp = intercept[0] + coefficients[0][0]*60 + coefficients[0][1]*2 + coefficients[0][2]*199
# p_num = pow(math.e, p_num_exp);
# p_den = 1 + p_num
# p = p_num/p_den
# print("Ejercicio C: La probabilidad de que tenga la enfermedad es: " + str(p) + " como p>0.5 esta enfermo\n")

# # d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
# data = file_data[['sex', 'age', 'cad.dur', 'choleste']]
# #data = normalize_data(data)
# data = data.to_numpy()


# d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
data = file_data[['sex', 'age', 'cad.dur', 'choleste']]
data = normalize_data(data)
data = data.to_numpy()

# train_percentage = 0.8
# train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

# print('Ejercicio D, con sex')
# logistic_training('d', train_data, test_data, train_labels, test_labels)

# e) 

data = file_data[['age', 'choleste', 'cad.dur']]
scaler = StandardScaler()
data1 = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
data1 = data1.to_numpy()

train_percentage = 0.05
train_data, test_data, train_labels, test_labels = train_test_split(data1, label, train_size=train_percentage)

# hc = HierarchicalClustering()
# root = hc.group(np.matrix(train_data))
# root
#
# for i in range(len(train_labels)):
#     print(f"{i}:\t{hc.binaryClassify(i)}\t{train_labels[i]}")


train_percentage = 0.9
train_data, test_data, train_labels, test_labels = train_test_split(data1, label, train_size=train_percentage)

# kn = KohonenNetwork(len(train_data[0]), 4, train_data)
# kn.train(np.matrix(train_data), 10000)
# kn



model = KMeans(3)
model.fit(train_data, train_labels)

m = model.predict(test_data)
m = np.array(m)
print(len(m))
correct = 0
for i in range(len(m)):
    if m[i] == test_labels[i]:
        correct += 1;

print(correct)

# model = K_Means(2)
# model.fit(train_data)
#
# m = model.predict(test_data)
# print(m)

# print(train_labels)
#label me dice si es 1 o 0, enfermo o no enfermo !!


# clf = clf.predict(test_data)
# print(clf)