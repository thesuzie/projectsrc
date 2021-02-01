import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords

from encode_simple import custom_tokenize, tfidf_encode
from evaluation import evaluate_classifier


def tfidf_model(train, balance):
    if balance == "class_weight":
        model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')), MultinomialNB(fit_prior=False))
        model.fit(train["Content Cleaned"], train["Label"])

    else:

        model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')), MultinomialNB())
        model.fit(train["Content Cleaned"], train["Label"])

    return model


def count_model(train, balance):
    if balance == "class_weight":
        model = make_pipeline(CountVectorizer(stop_words=stopwords.words('english')), MultinomialNB(fit_prior=False))
        model.fit(train["Content Cleaned"], train["Label"])
    else:
        model = make_pipeline(CountVectorizer(stop_words=stopwords.words('english')), MultinomialNB())
        model.fit(train["Content Cleaned"], train["Label"])

    return model


def main():
    try:
        encode = sys.argv[1]
        train = pd.read_csv(sys.argv[2], sep=',', header=0)
        test = pd.read_csv(sys.argv[3], sep=',', header=0)
        balance = sys.argv[4]

    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <encoding_method> <train_file> <test_file> <balance> ")

    if encode == "tfidf":
        classifier = tfidf_model(train, balance)
    else: # encode == "count":
        classifier = count_model(train, balance)

    # Predictions on test set
    pred = classifier.predict(test["Content Cleaned"])
    np.savetxt(f'./output/y_NB_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/NB_{encode}_{balance}.png")

    return None


if __name__ == "__main__":
    main()
