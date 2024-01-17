import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Receiver Operating Characteristic (ROC) curves for a multiclass classification model. 
# ROC curves are used to evaluate the performance of binary classification models and are 
# extended to the multiclass case here.
def plot_multiclass_roc_func(clf, X_test, y_test, n_classes, figsize=(5,5)):
    # clf: The classifier model for which you want to plot ROC curves.
    # X_test: The test dataset features.
    # y_test: The true labels for the test dataset.
    # n_classes: The number of classes in your multiclass classification problem.
    # figsize: The size of the resulting plot (default is (5,5)).

    # computes the decision scores (sometimes called confidence scores) for each class. 
    # The decision function returns a score for each class, and these scores are used to 
    # create ROC curves.
    y_score = clf.decision_function(X_test)

    # Dictionary to store rates for each class:

    # False Positive Rates
    fpr = dict()
    # True Positive Rates
    tpr = dict()
    # Area Under the Curve (AUC) 
    roc_auc = dict()

    # calculate dummies once
    # This converts the true labels y_test into one-hot encoded format, 
    # which is necessary for computing ROC curves for each class.
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()