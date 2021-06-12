#%% Libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# %% confmatrix
def twoclass(y_true, y_pred_proba, threshold=0.5, classnames=None, normalize=False, title='', cmap=plt.cm.Blues, showfig=True, verbose=3):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = confusion_matrix(y_true, y_pred_proba>=threshold)
    if isinstance(classnames, type(None)):
        classnames = ['Class1','Class2']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose>=3: print("Normalized confusion matrix")
    else:
        if verbose>=3: print('Confusion matrix, without normalization')

    if verbose>=3:
        print(cm)
	
    if showfig:
        makeplot(cm, classnames=classnames, title=title, normalize=normalize, cmap=cmap)

    return(cm)

#%%
def makeplot(cm, classnames=None, title='', normalize=False, cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + 'Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=45)
    plt.yticks(tick_marks, classnames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Network 1')
        plt.xlabel('Network 2')
        plt.tight_layout()
        plt.grid(False)