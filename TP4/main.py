import pandas as pd
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

# fills values that are Nan
def fill_data(fill):
    df = fill.sort_values(by=['age'])
    df.fillna(inplace=True, method='ffill')
    return df;

# import data
file_data = pd.read_csv('data/acath.csv', sep=';')
file_data = fill_data(file_data);
data = file_data[['age', 'cad.dur', 'choleste']]
label = file_data['sigdz'].to_list()


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# normalize data
aux = data.values;
min_max_scaler = preprocessing.MinMaxScaler();
x_scaled = min_max_scaler.fit_transform(aux);
data = pd.DataFrame(x_scaled);

data = data.to_numpy();

# train_test_split taken from sklearn.model_selection
train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage);
print(train_data)


# trainig Logistic taken from sklearn.linear_model
def logistic_training(train_data, test_data, train_labels, test_labels):
    train = LogisticRegression(n_jobs=3, C=0.3);
    train.fit(train_data, train_labels)
    #testing results
    predictions = train.predict(test_data)
    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])

    # print a nice matrix
    plot_confusion_matrix(cm, target_names=['Negative', 'Positive'], title='Logistic Regression')

logistic_training(train_data, test_data, train_labels, test_labels)

# d) tenemos que agregar el sexo y hacemos el mismo procedimiento que (a)
data = file_data[['sex', 'age', 'cad.dur', 'choleste']]

aux = data.values;
min_max_scaler = preprocessing.MinMaxScaler();
x_scaled = min_max_scaler.fit_transform(aux);
data = pd.DataFrame(x_scaled);

data = data.to_numpy()

train_percentage = 0.8
train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=train_percentage)

logistic_training(train_data, test_data, train_labels, test_labels)