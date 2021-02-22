import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from encode import nn_encode
from keras.preprocessing.sequence import pad_sequences


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])

    return None


train_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/train_IM_no8.csv")
test_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/test_IM_no8.csv")

train_inds, test_inds = nn_encode(train_dataset, test_dataset)

train_inds.tofile("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_train.csv", sep=",", format="%s")
test_inds.tofile("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_test.csv", sep=",", format="%s")


def find_longest_sequence(inds, longest_seq):
    '''find the longest sequence in the dataframe'''
    for i in inds:
        seqlen = len(i)
        if seqlen > longest_seq:  # update high water mark if new longest sequence encountered
            longest_seq = seqlen
    return longest_seq


train_longest = find_longest_sequence(train_inds, 0)
print('The longest sequence in the training set is %i tokens long' % train_longest)
print(train_inds[0])

seq_length = 273
