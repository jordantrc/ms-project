# analysis.py

import numpy as np
import itertools
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def confusion_matrix(predictions, labels, classes):
    """
    produces and returns a confusion matrix given the predictions generated by
    tensorflow (in one-hot format), and string labels.
    """
    #print("pred = %s, type = %s, labels = %s, type = %s, classes = %s, type = %s" % (predictions, type(predictions), labels, type(labels), classes, type(classes)))

    y_true = []
    y_pred = []

    for p in predictions:
    	y_pred.append(p[0])

    for l in labels:
    	y_true.append(l[0])

    cm = metrics.confusion_matrix(y_true, y_pred, classes)

    return cm


def plot_confusion_matrix(cm, classes, filename,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    #print(cm)

    #thresh = cm.max() * 0.73
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, "{0:.4f}".format(cm[i, j]),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(filename)
    plt.gcf().clear()
    plt.cla()
    plt.clf()
    plt.close()