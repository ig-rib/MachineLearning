from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# This kind of normalization can be interesting in case of class imbalance to have a more visual
# interpretation of which class is being misclassified

def plot_confusion_matrix_bis(train, predictions, test_data, test_labels, normalize='true'):
    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    if normalize == 'None':
        plot_confusion_matrix(train, test_data, test_labels)
    else:
        plot_confusion_matrix(train, test_data, test_labels, normalize=normalize)

    target_names = ['Negativo', 'Positivo']
    tick_marks = np.arange(len(target_names))

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclassified = 1 - accuracy

    print("accuracy: " + str(accuracy))
    print("misclassified: " + str(misclassified)+"\n")

    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.title('Regresión Logística')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, misclassified))
    plt.ylabel('Original')
    plt.show()
