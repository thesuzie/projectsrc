from sklearn.pipeline import make_pipeline

from encode_simple import custom_tokenize, tfidf_encode
import sys
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_classifier


def tfidf_model(train, balance=None):

    if balance == "class_weight":
        print(balance)
        model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')),
                              LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',
                                                 solver='newton-cg', class_weight='balanced'))
        model.fit(train["Content Cleaned"], train["Label"])
    else:
        model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')),
                          LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',
                                             solver='newton-cg'))
        model.fit(train["Content Cleaned"], train["Label"])
    return model


def count_model(train, balance=None):
    if balance == "class_weight":
        print(balance)
        model = make_pipeline(CountVectorizer(stop_words=stopwords.words('english')),
                              LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',
                                                 solver='newton-cg', class_weight='balanced'))
        model.fit(train["Content Cleaned"], train["Label"])
    else:
        model = make_pipeline(CountVectorizer(stop_words=stopwords.words('english')),
                          LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',
                                             solver='newton-cg'))
        model.fit(train["Content Cleaned"], train["Label"])

    return model


def main():
    try:
        encode = sys.argv[1]
        train = pd.read_csv(sys.argv[2], sep=',', header=0)
        test = pd.read_csv(sys.argv[3], sep=',', header=0)
        balance = sys.argv[4]

    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <encoding_method> <train_file> <test_file> <balance>")

    # Building the model

    if encode == "tfidf":
        vec = TfidfVectorizer(stop_words=stopwords.words('english'))
        #classifier = tfidf_model(train, balance)
    else:
        vec = CountVectorizer(stop_words=stopwords.words('english'))
        #classifier = count_model(train, balance)

    if balance == "class_weight":
        model = make_pipeline(vec, LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg', class_weight='balanced'))
    elif balance == "rand_ov_samp":
        model = make_pipeline(vec, RandomOverSampler(random_state=0), LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg'))
    else:
        model = make_pipeline(vec, LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg'))

    model.fit(train["Content Cleaned"], train["Label"])

    # make predictions
    pred = model.predict(test["Content Cleaned"])

    np.savetxt(f'./output/y_LogReg_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/LogReg_{encode}_{balance}.png")

    return None

if __name__ == "__main__":
    main()