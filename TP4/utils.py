from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# This kind of normalization can be interesting in case of class imbalance to have a more visual
# interpretation of which class is being misclassified

def plot_confusion_matrix_bis(title, train, predictions, test_data, test_labels, normalize='None'):
    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    if normalize == 'None':
        plot_confusion_matrix(train, test_data, test_labels)
    else:
        plot_confusion_matrix(train, test_data, test_labels, normalize=normalize)

    target_names = ['No enfermo', 'Enfermo']
    tick_marks = np.arange(len(target_names))

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclassified = 1 - accuracy

    print("accuracy: " + str(accuracy))
    print("misclassified: " + str(misclassified))
    print(cm)
    print("\n")

    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.title(title)
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, misclassified))
    plt.ylabel('Original')
    plt.show()


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
