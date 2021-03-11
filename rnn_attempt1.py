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
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Input, Bidirectional, Dropout, GlobalMaxPool1D
from keras.callbacks import EarlyStopping
from evaluation import evaluate_classifier

# values from the full (unbalanced) dataset



LABELS = ["sex", "relationships", "ewhoring", "online_crime", "description", "real_world_abuse",
          "politics_ideology", "story"]
context_categories = ["1", "2", "3", "4", "5", "6", "7", "9"]
max_seq_len = 273
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
    padded = pad_sequences(sequences, maxlen=max_seq_len, dtype='int32', padding='post',
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


    #categories = ["0", "sex", "relationships", "ewhoring", "online_crime", "description", "real_world_abuse", "politics_ideology", "8", "story"]

    preds = predictions
    labels = []
    for p in preds:
        index_max = (np.where(p == np.amax(p)))[0]
        #print(f"index:  {index_max}")
        l = LABELS[index_max[0]]
        labels.append(l)

    return labels


def label_to_array(labels):
    arrs = [np.array(l) for l in labels]
    return arrs


def create_validation_set(x,y):
    X_train, X_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=123,
                                                      test_size=0.1)

    return X_train, X_val, y_train, y_val


def make_model():

    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, embedding_dim)(inp)
    x = LSTM(60, return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    # removed bias from dense layer (was not working...)
    x = Dense(n_labs, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model



train_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_train.csv")
test_dataset = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/rnn_test.csv")

x_train = create_padded(convert_to_list(train_dataset["Token Indices"]))
y_train = np.array(onehot_labels(train_dataset["Label"]))
print(y_train[0:4])

#weights = find_biases(train_dataset["Label"])

X_train, X_val, y_train, y_val = create_validation_set(x_train, y_train)

x_test = create_padded(convert_to_list(test_dataset["Token Indices"]))
y_test = np.array(onehot_labels(test_dataset["Label"]))

model = make_model()
batch_size = 32
epochs = 2

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [early]


model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_list)

y_pred = model.predict(x_test)

print(y_pred[0:4])


np.savetxt("/Users/suziewelby/year3/compsci/project/src/output/y_rnn_indicies_IM.csv", y_pred, delimiter=',')

evaluate_classifier(y_test, single_pred(y_pred), "/Users/suziewelby/year3/compsci/project/src/output/rnn_indices_IM.png")
