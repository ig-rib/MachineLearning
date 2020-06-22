#!/bin/python3

import pandas as pd
from utils import plot_confusion_matrix_bis
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

from sklearn.model_selection import train_test_split
train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio B, sin sex')
coefficients, intercept = logistic_training('b', train_data, test_data, train_labels, test_labels)

p_num_exp = intercept[0] + coefficients[0][0]*60 + coefficients[0][1]*2 + coefficients[0][2]*199
p_num = pow(math.e, p_num_exp);
p_den = 1 + p_num
p = p_num/p_den
print("Ejercicio C: La probabilidad de que tenga la enfermedad es: " + str(p) + " como p>0.5 esta enfermo\n")

# d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
data = file_data[['sex', 'age', 'cad.dur', 'choleste']]
#data = normalize_data(data)
data = data.to_numpy()

train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

print('Ejercicio D, con sex')
logistic_training('d', train_data, test_data, train_labels, test_labels)

# # e) 

def print_unsupervised_confusion_matrix(title, classifiedExamples, actual):
    
    TP = len([True for index, x in enumerate(classifiedExamples) if classifiedExamples[index] == actual[index] and classifiedExamples[index] == 0])
    TN = len([True for index, x in enumerate(classifiedExamples) if classifiedExamples[index] == actual[index] and classifiedExamples[index] == 1])
    FP = len([True for index, x in enumerate(classifiedExamples) if actual[index] == 0 and classifiedExamples[index] == 1])
    FN = len([True for index, x in enumerate(classifiedExamples) if actual[index] == 1 and classifiedExamples[index] == 0])

    print(f"{title}:\n\tAc/Pr\tN\tP\n\tN\t{TN}\t{FP}\n\tP\t{FN}\t{TP}\n\n")

data = file_data[['age', 'choleste', 'cad.dur']]
scaler = StandardScaler()
data1 = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
data1 = data1.to_numpy()

train_percentage = 0.05
train_data, test_data, train_labels, test_labels = train_test_split(data1, label, train_size=train_percentage)

###########################
### Hierarchical Clustering
###########################


hc = HierarchicalClustering()
root = hc.group(np.matrix(train_data))

classifiedExamples = [hc.binaryClassify(i) for i in range(len(train_data))]


Acount = len([x for x in classifiedExamples if x == 'A'])
bestClass = max([('A', Acount), ('B', len(classifiedExamples) - Acount)], key=lambda x: x[1])[0]
classification = {'A': None, 'B': None}

classification[bestClass] = max( [
                            (0, len([x for index, x in enumerate(classifiedExamples) if x == bestClass and train_labels[index] == 0])), 
                            (1, len([x for index, x in enumerate(classifiedExamples) if x == bestClass and train_labels[index] == 1]))
                            ],
                            key=lambda x: x[1]
                            )[0]
classification['B' if bestClass == 'A' else 'A'] = 1 if classification['A'] == 0 else 0

classifiedExamples[:] = [classification[x] for x in classifiedExamples]

print_unsupervised_confusion_matrix('Hierarchical Clustering', classifiedExamples, train_labels)

##########################
## Kohonen
##########################

train_percentage = 0.9
train_data, test_data, train_labels, test_labels = train_test_split(data1, label, train_size=train_percentage)

kn = KohonenNetwork(len(train_data[0]), 4, D=train_data)

classZeroNodes = [train_data[i] for i, data in enumerate(train_data) if train_labels[i] == 0]
classOneNodes = [train_data[i] for i, data in enumerate(train_data) if train_labels[i] == 1]

kn.W = np.asmatrix([
    classZeroNodes[0],
    classZeroNodes[1],
    classOneNodes[0],
    classOneNodes[1]
])

kn.train(np.matrix(train_data), 25000, R=1)


# classifiedExamples = ['A' if kn.getClass(train_data[i])[0] == 0 else 'B' for i in range(len(train_data))]
classifiedExamples = [kn.getClass(train_data[i])[0] for i in range(len(train_data))]

print_unsupervised_confusion_matrix('Kohonen Nets - Simplified', classifiedExamples, train_labels)
