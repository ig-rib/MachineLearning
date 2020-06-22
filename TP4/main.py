#!/bin/python3

import pandas as pd
from utils import plot_confusion_matrix_bis, fill_data, normalize_data
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# for confusion matrix printing
import seaborn as sns
import matplotlib.pyplot as plt


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
#data = normalize_data(data)
data = data.to_numpy()

# from sklearn.model_selection import train_test_split
train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio B, sin sex')
coefficients, intercept = logistic_training('b', train_data, test_data, train_labels, test_labels)

p_num_exp = intercept[0] + coefficients[0][0]*60 + coefficients[0][1]*2 + coefficients[0][2]*199
p_num = pow(base=math.e, exp=p_num_exp);
p_den = 1 + p_num
p = p_num/p_den
print("Ejercicio C: La probabilidad de que tenga la enfermedad es: " + str(p) + " como p>0.5 esta enfermo\n")


# d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
data = file_data[['sex', 'age', 'cad.dur', 'choleste']]
data = normalize_data(data)
data = data.to_numpy()

train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio D, con sex')
logistic_training('d', train_data, test_data, train_labels, test_labels)

