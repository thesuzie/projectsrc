import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np


# todo: create classifier evaluation function that prints out all needed values


# this should be in a different file - not really relevant here
def combine_annotations(file1, file2):
    # INPUT: file1, file2 dataframes
    # OUTPUT: new dataframe with labels from file 2 in label1 sent1 columns and final columns
    combined = file1
    combined['Label 1'] = file2['Label']
    combined['Sentiment 1'] = file2['Sentiment']

    final_l = [0] * len(file1['Label'])
    final_s = [0] * len(file1['Label'])
    i = 0
    while i < len(file1['Label']):
        if combined['Label'][i] == combined['Label 1'][i]:
            final_l[i] = combined['Label'][i]
        else:
            final_l[i] = "X"

        if combined['Sentiment'][i] == combined['Sentiment 1'][i]:
            final_s[i] = combined['Sentiment'][i]
        else:
            final_s[i] = "X"
        i += 1

    combined["Final Label"] = final_l

    combined['Final Sentiment'] = final_s

    return combined


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
