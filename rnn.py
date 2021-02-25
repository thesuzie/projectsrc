import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from encode import nn_encode, create_token_vocab
from keras.preprocessing.sequence import pad_sequences

# values from the full (unbalanced) dataset
seq_length = 273
unique_tokens = 11942
oov = unique_tokens + 1
pad_value = 0.0
vocab_size = unique_tokens + 2
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])

    return None


def prep_indices():
    train_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/train_IM_no8.csv")
    test_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/test_IM_no8.csv")

    train_inds, test_inds = nn_encode(train_dataset, test_dataset)

    train_dataset["Token Indices"] = train_inds
    test_dataset["Token Indices"] = test_inds

    test_dataset.to_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_test.csv", index=True)
    train_dataset.to_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_train.csv", index=True)

    return None


def find_longest_sequence(inds, longest_seq):

    '''find the longest sequence in the dataframe'''
    for i in inds:
        seqlen = len(i)
        if seqlen > longest_seq:  # update high water mark if new longest sequence encountered
            longest_seq = seqlen
    return longest_seq


def create_padded(sequences):

    padded = pad_sequences(sequences, maxlen=seq_length, dtype='int32', padding='post',
                           truncating='post', value=pad_value)

    return padded


train_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_train.csv")
test_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_test.csv")

X_train = create_padded(train_dataset["Token Indices"].tolist())
y_train = train_dataset["Label"]

X_test = create_padded(test_dataset["Token Indices"])
y_test = test_dataset["Label"]

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64, # can tweak this value
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit((X_train,y_train), epochs=10,
                    validation_data=(X_test, y_test),
                    validation_steps=30)

test_loss, test_acc = model.evaluate((X_test, y_test))

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
plot_graphs(history, 'loss')
plt.ylim(0,None)




