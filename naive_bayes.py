import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from encode_simple import custom_tokenize


# TODO:
# split into different sets


def main():
    # need to update to make usage Usage: {sys.argv[0]} <train_file> <test_file>
    try:
        file = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <data_file>")

    data = pd.read_csv(file, header=0, sep=',')

    # would make a call to the function for each of the training and test sets
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)

    tfidf = vectorizer.fit_transform(data['Content Cleaned'][0:100])

    # could move this into the tfidf function if want to print
    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    # print(df)

    context_categories = np.unique(data['Label'][0:100])
    # print(context_categories)

    # would have input as test train already so would not split here
    x_train, x_test, y_train, y_test = train_test_split(tfidf.toarray(), data['Label'][0:100])
    # print(f"x_train: {x_train}")
    # print(f"x_test: {x_test}")

    # print(f"y_test: {y_train}")

    # Building the model
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    # Predictions on test set
    pred = classifier.predict(x_test)

    print(confusion_matrix(y_test, pred))
    # need to use better metrics
    print(accuracy_score(y_test, pred))

    mat = confusion_matrix(y_test, pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=context_categories,
                yticklabels=context_categories)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    print("The accuracy is {}".format(accuracy_score(y_test, pred)))


if __name__ == "__main__":
    main()
