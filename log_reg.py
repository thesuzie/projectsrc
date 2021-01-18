from encode_simple import custom_tokenize, tfidf_encode
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def main():

    try:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <train_file> <test_file>")

    train = pd.read_csv(train_file, header=0, sep=',')
    encode_train = tfidf_encode(train)

    test = pd.read_csv(test_file,header=0, sep=',')
    encode_test = tfidf_encode(test)

    context_categories = [1,2,3,4,5,6,7,8,9]
    # print(context_categories)

    x_train = encode_train.toarray()
    y_train = train['Label']

    x_test = encode_test.toarray()
    y_test = test['Label']

    # print(f"x_train: {x_train}")
    # print(f"x_test: {x_test}")
    # print(f"y_test: {y_train}")

    # Building the model
    classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(x_train, y_train)

    # Predictions on test set
    preds = classifier.predict(x_test)

    print(confusion_matrix(y_test, preds))
    # need to use better metrics
    print(accuracy_score(y_test, preds))

    mat = confusion_matrix(y_test, preds)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=context_categories,
                yticklabels=context_categories)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    print("The accuracy is {}".format(accuracy_score(y_test, pred)))


if __name__ == "__main__":
    main()