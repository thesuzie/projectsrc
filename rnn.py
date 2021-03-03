import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import ast
from encode import nn_encode, create_token_vocab
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# sklearn libraries
from sklearn.model_selection import train_test_split
# keras libraries
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Input

# values from the full (unbalanced) dataset
seq_length = 273
unique_tokens = 11942
oov = unique_tokens + 1
pad_value = 0.0
vocab_size = unique_tokens + 2
embedding_dim = 64
n_labs = 8
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

# prep_indices()

LABELS = ["sex", "relationships", "ewhoring", "online_crime", "description", "real_world_abuse",
          "politics_ideology", "story"]
seq_length = 273
unique_tokens = 11942
oov = unique_tokens + 1
pad_value = 0.0
vocab_size = unique_tokens + 2
embedding_dim = 64
n_labs = 8
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

# prep_indices()

LABELS = ["sex", "relationships", "ewhoring", "online_crime", "description", "real_world_abuse",
          "politics_ideology", "story"]


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

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
    # print(f"first sequence: {sequences[0]}")
    padded = pad_sequences(sequences, maxlen=seq_length, dtype='int32', padding='post',
                           truncating='post', value=pad_value)

    return padded


def find_biases(labels):
    all_labs = labels
    label_count = Counter(all_labs)
    total_labs = len(all_labs)
    print(label_count)
    print(total_labs)

    # use this to define an initial model bias
    initial_bias = [(label_count[1] / total_labs),
                    (label_count[2] / total_labs), (label_count[3] / total_labs), (label_count[4] / total_labs),
                    (label_count[5] / total_labs),
                    (label_count[6] / total_labs), (label_count[7] / total_labs), (label_count[9] / total_labs)]
    print('Initial bias:')
    print(initial_bias)

    return initial_bias


def onehot_labels(labels):
    length = len(labels)
    encoded = [[0]] * length
    i = 0
    while i < length:
        if labels[i] == 1:
            encoded[i] = [1, 0, 0, 0, 0, 0, 0, 0]
        elif labels[i] == 2:
            encoded[i] = [0, 1, 0, 0, 0, 0, 0, 0]
        elif labels[i] == 3:
            encoded[i] = [0, 0, 1, 0, 0, 0, 0, 0]
        elif labels[i] == 4:
            encoded[i] = [0, 0, 0, 1, 0, 0, 0, 0]
        elif labels[i] == 5:
            encoded[i] = [0, 0, 0, 0, 1, 0, 0, 0]
        elif labels[i] == 6:
            encoded[i] = [0, 0, 0, 0, 0, 1, 0, 0]
        elif labels[i] == 7:
            encoded[i] = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            encoded[i] = [0, 0, 0, 0, 0, 0, 0, 1]

        i += 1

    return encoded



def convert_to_list(tokens):
    i = 0
    l = len(tokens)
    tok_list = [[0]] * l
    # print(l)
    while i < l:
        tok_list[i] = ast.literal_eval(tokens[i])
        i += 1
    # print(tok_list[0])
    return tok_list


def single_pred(predictions):
    single_pred = [np.where(preds == np.amax(preds)) for preds in predictions]

    labels =[]
    for p in single_pred:
        i = p[0][0]
        labels.append(LABELS[i])
    return labels


def label_to_array(labels):
    arrs = [[l] for l in labels]
    return arrs



train_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_train.csv")
test_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_test.csv")
#
# X_train_data = create_padded(convert_to_list(train_dataset["Token Indices"]))
# y_train_data = label_to_array(train_dataset["Label"])
# X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, shuffle=True, random_state=123, test_size=0.1)
#
# X_test = create_padded(convert_to_list(test_dataset["Token Indices"]))
# y_test = label_to_array(test_dataset["Label"])
#
# print(y_train[9])
# print(y_test[5])
#
# model = tf.keras.Sequential([
#
#     tf.keras.layers.Embedding(vocab_size, embedding_dim),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#     tf.keras.layers.Dense(embedding_dim, activation='relu'),
#     tf.keras.layers.Dense(n_labs, activation='softmax')
# ])
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# num_epochs = 10
# history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2)
#
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

## ONE HOT ENCODED LABELS ATTEMTPT (METHOD TO PICK ONE LABEL NOT WORKING)
# X_train_data = create_padded(convert_to_list(train_dataset["Token Indices"]))
# y_train_data_onehot = onehot_labels(train_dataset["Label"])
# y_train_data = pd.DataFrame(y_train_data_onehot, columns=LABELS)
# # bias = find_biases(y_train_data)
#
#
# X_test = create_padded(convert_to_list(test_dataset["Token Indices"]))
# y_test_onehot = onehot_labels(test_dataset["Label"])
#
# y_test = pd.DataFrame(y_test_onehot, columns=LABELS)
#
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='AUC', verbose=1, patience=10, mode='max', restore_best_weights=True)
#
#
# rnn_model = Sequential()
# rnn_model.add(Embedding(vocab_size, 128))
# rnn_model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
# rnn_model.add(Dense(units=8, activation="sigmoid"))
#
# print(rnn_model.summary())
#
# rnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])
#
# X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, shuffle=True, random_state=123, test_size=0.1)
#
# history = rnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])
#
# y_preds = rnn_model.predict(X_test)
y_preds = [[0.09796715, 0.14062464, 0.27032346, 0.03484038, 0.1296387,  0.0384523,
  0.06302279 ,0.21262997],
 [0.09796715, 0.14062464, 0.27032346 ,0.03484038 ,0.1296387 , 0.0384523
 , 0.06302282, 0.21262997],
 [0.09796715 ,0.14062467 ,0.27032346, 0.03484043 ,0.1296387,  0.0384523,
  0.06302282, 0.21262997],
 [0.09796715, 0.14062464, 0.27032346, 0.03484043, 0.1296387,  0.0384523,
  0.06302279, 0.21262997]
, [0.09796715, 0.14062464, 0.27032346, 0.03484043 ,0.1296387 , 0.0384523,
  0.06302282, 0.21262997]]
#print(y_preds[0:5])

predicted_word = single_pred(y_preds)
actual_word = test_dataset["Label"]

print(f"predicted: {predicted_word[0:15]}")

print(f"actual: {actual_word[0:15]}")

#plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")
#


# ORIGINAL ATTEMPT
# model = tf.keras.Sequential([
#
#     tf.keras.layers.Embedding(
#         input_dim=vocab_size,
#         output_dim=64, # can tweak this value
#         # Use masking to handle the variable sequence lengths
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(n_labs, activation='relu', bias_initializer=tf.keras.initializers.Constant(bias))
#
# ])
#
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=METRICS)
#
#
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='accuracy', verbose=1, patience=10, mode='max', restore_best_weights=True)
#
# EPOCHS = 50
# BATCH_SIZE = 30
#
# print("fitting...")
#
# history = model.fit(X_train, y_train, epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[early_stopping]
#                     )
#
# print("finished fitting")
#
# preds = np.argmax(model.predict(X_test), axis=-1)
# #flat_preds = [p for pred in preds for p in pred]
# print(Counter(preds))
#
# test_loss, test_acc = model.evaluate((X_test, y_test))
#
#
# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))
#
# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None,1)
# plt.subplot(1,2,2)
# plot_graphs(history, 'loss')
# plt.ylim(0,None)
# plt.show()
#
#
#
