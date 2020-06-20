#!/bin/python3

import pandas as pd
from utils import plot_confusion_matrix_bis
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

# DATA MANAGMENT

def fill_data(fill):
    # fill values that are Nan
    d = fill.sort_values(by=['age'])
    d.fillna(inplace=True, method='ffill')
    return d;


def normalize_data(d):
    aux = d.values;
    s = preprocessing.MinMaxScaler();
    x = s.fit_transform(aux);
    return pd.DataFrame(x);


def logistic_training(ex, train_data, test_data, train_labels, test_labels):
    # trainig Logistic taken from sklearn.linear_model
    train = LogisticRegression(n_jobs=3, C=0.3);
    train.fit(train_data, train_labels)

    # testing results
    predictions = train.predict(test_data)
    title = 'Regresión Logística'
    if ex == 'b':
        title = title + ' sin sex (b)'
    if ex == 'd':
        title = title + ' con sex (d)'
    plot_confusion_matrix_bis(title, train, predictions, test_data, test_labels, normalize='true')


# import data
file_data = pd.read_csv('data/acath.csv', sep=';')
file_data = fill_data(file_data)
data = file_data[['age', 'cad.dur', 'choleste']]
label = file_data['sigdz'].to_list()

# normalize data
data = normalize_data(data)
data = data.to_numpy()

# from sklearn.model_selection import train_test_split
train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio b, sin sex')
logistic_training('b', train_data, test_data, train_labels, test_labels)

# d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
data = file_data[['sex', 'age', 'cad.dur', 'choleste']]
data = normalize_data(data)
data = data.to_numpy()

train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio d, con sex')
logistic_training('d', train_data, test_data, train_labels, test_labels)
