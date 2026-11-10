"""confmatrix."""
# ------------------------------------
# Name        : confmatrix.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

import logging
logger = logging.getLogger("bnlearn")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix
import numpy as np

# %% confmatrix


def twoclass(y_true, y_pred_proba, threshold=0.5, classnames=None, normalize=False, title='', cmap=plt.cm.Blues, showfig=True, xlabel='Network 2', ylabel='Network 1'):
    """Compute and plot a two-class confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1 or bool).
    y_pred_proba : array-like
        Predicted labels or probabilities. Values >= threshold are positives.
    threshold : float
        Decision threshold.
    classnames : list of str
        Names of the two classes, e.g., ['Disconnected', 'Connected'].
    normalize : bool
        True: show row-normalized fractions instead of counts. The returned matrix is normalized too.
    title : str
        Prefix of the figure title.
    showfig : bool
        Show the confusion matrix.
    xlabel, ylabel : str
        Axis labels (x: predicted/second network, y: true/first network).

    Returns
    -------
    np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]] (counts, or fractions when normalize=True).
    """
    # labels=[False, True] guarantees a 2x2 matrix, also when one of the classes is absent
    # (e.g., a network without any edges).
    counts = confusion_matrix(np.asarray(y_true).astype(bool), np.asarray(y_pred_proba >= threshold).astype(bool), labels=[False, True])
    if isinstance(classnames, type(None)):
        classnames = ['Class1', 'Class2']

    cm = counts
    if normalize:
        cm = counts.astype('float') / np.maximum(counts.sum(axis=1)[:, np.newaxis], 1)
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')
    logger.info(cm)
    if showfig:
        makeplot(cm, classnames=classnames, title=title, normalize=normalize, cmap=cmap, xlabel=xlabel, ylabel=ylabel, counts=counts)

    return(cm)

# %% Scores


def _scores(counts):
    """Precision, recall and F1 of the positive class from a 2x2 matrix of counts."""
    tn, fp, fn, tp = np.asarray(counts).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    return precision, recall, f1

# %% Plot


def makeplot(cm, classnames=None, title='', normalize=False, cmap=plt.cm.Blues, xlabel='Network 2', ylabel='Network 1', counts=None, figsize=(7, 6)):
    """Plot the confusion matrix.

    - The cell shading is the share of each row (the true class). A raw-count colorscale is dominated by the
      'true negative' cell (all disconnected node pairs) and makes the other cells unreadable.
    - Each cell shows the count, its name (TN/FP/FN/TP) and the row percentage.
    - The tick labels show the number of samples per class.
    - The title shows precision, recall and F1 of the positive class.
    - FP and FN cells are outlined like the edges in network.compare (FP: red, FN: black dashed).
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    if classnames is None:
        classnames = ['Class%d' % (i + 1) for i in range(n)]
    counts = cm if counts is None else np.asarray(counts)

    row_tot = counts.sum(axis=1, keepdims=True)
    col_tot = counts.sum(axis=0)
    share = np.divide(counts, row_tot, out=np.zeros(counts.shape, dtype=float), where=row_tot > 0)
    names = [['TN', 'FP'], ['FN', 'TP']] if n == 2 else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(share, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            color = 'white' if share[i, j] > 0.5 else 'black'
            main = ('%.2f' % cm[i, j]) if normalize else ('{:,}'.format(int(counts[i, j])))
            ax.text(j, i - 0.08, main, ha='center', va='center', color=color, fontsize=18, fontweight='bold')
            sub = ('%s  ' % names[i][j] if names else '') + '{:.0%}'.format(share[i, j])
            ax.text(j, i + 0.2, sub, ha='center', va='center', color=color, fontsize=10)

    if names is not None:
        ax.add_patch(Rectangle((0.5 + 0.03, -0.5 + 0.03), 0.94, 0.94, fill=False, edgecolor='red', linewidth=3))
        ax.add_patch(Rectangle((-0.5 + 0.03, 0.5 + 0.03), 0.94, 0.94, fill=False, edgecolor='black', linewidth=2.5, linestyle='--'))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(['%s\n(n=%s)' % (classnames[j], '{:,}'.format(int(col_tot[j]))) for j in range(n)])
    ax.set_yticklabels(['%s\n(n=%s)' % (classnames[i], '{:,}'.format(int(row_tot[i, 0]))) for i in range(n)])
    ax.set_xlabel(xlabel, labelpad=10, fontsize=11)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    heading = (title + ' ' if title else '') + 'Confusion matrix'
    if n == 2:
        precision, recall, f1 = _scores(counts)
        fmt = lambda x: 'n/a' if np.isnan(x) else '%.2f' % x
        heading += '\nPrecision %s  |  Recall %s  |  F1 %s' % (fmt(precision), fmt(recall), fmt(f1))
    ax.set_title(heading, fontsize=12, pad=14)
    fig.text(0.5, 0.01, 'Percentages are the share of each row (true class).', ha='center', fontsize=8, color='gray')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
