import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler, SMOTE

from encode_simple import custom_tokenize, tfidf_encode, count_encode
from evaluation import evaluate_classifier



def main():
    try:
        encode = sys.argv[1]
        train = pd.read_csv(sys.argv[2], sep=',', header=0)
        test = pd.read_csv(sys.argv[3], sep=',', header=0)
        balance = sys.argv[4]

    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <encoding_method> <train_file> <test_file> <balance> ")

    # Building the model
    if encode == "tfidf":
        vec = tfidf_encode(train)
    else:
        vec = count_encode(train)

    X_encoded = vec.transform(train["Content Cleaned"])

    if balance == "rand_ov_samp":
        X, y = RandomOverSampler(random_state=0).fit_resample(X_encoded, train["Label"])
        classifier = MultinomialNB()

    elif balance == "smote":
        X, y = SMOTE().fit_resample(X_encoded, train["Label"])
        classifier = MultinomialNB()

    else:  # balance == "class_weight"
        classifier = MultinomialNB()
        X = X_encoded
        y = train["Label"]

    classifier.fit(X, y)

    X_test = vec.transform(test["Content Cleaned"])

    pred = classifier.predict(X_test)

    np.savetxt(f'./output/y_NB_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/NB_{encode}_{balance}.png")

    return None


if __name__ == "__main__":
    main()
