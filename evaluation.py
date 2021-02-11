import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_classifier(y, preds, file):
    context_categories = [1, 2, 3, 4, 5, 6, 7, 9]

    print(classification_report(y, preds, context_categories))

    mat = confusion_matrix(y, preds)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=context_categories,
                yticklabels=context_categories)
    plt.xlabel("True labels")
    plt.ylabel("Predicted label")
    plt.savefig(file)

    return None


def count_labels(txt):
    # INPUT: dataframe with label column
    # OUTPUT: NONE, func prints answer
    (unique, counts) = np.unique(txt['Label'], return_counts=True)
    print('label, Count of labels')
    print(np.asarray((unique, counts)).T)
    return None


def count_sent(txt):
    # INPUT: dataframe with sentiment column
    # OUTPUT: NONE, func prints answer
    (unique, counts) = np.unique(txt['Sentiment'], return_counts=True)
    print('label, Count of Sentiment')
    print(np.asarray((unique, counts)).T)


"""
file1 = pd.read_csv('./data/data1annotatedsgw.csv')
file2 = pd.read_csv('./data/data1annotatedildi.csv')

print("File 1:")
count_labels(file1)
count_sent(file1)
print("File 2:")
count_labels(file2)
count_sent(file2)

combined = combine_annotations(file1, file2)

combined.to_csv('./data/data1combined.csv')

kappa_label = cohen_kappa_score(combined['Label'], combined['Label 1'])
print(f"Cohen's kappa for class: {kappa_label}")

kappa_sent = cohen_kappa_score(combined['Sentiment'], combined['Sentiment 1'])
print(f"Cohen's kappa for sentiment: {kappa_sent}")
"""
