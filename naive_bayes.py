import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from encode_simple import custom_tokenize, tfidf_encode
from evaluation import evaluate_classifier


def main():
    try:
        X_train = pd.read_csv(sys.argv[1], sep=',', header=0)
        X_test = pd.read_csv(sys.argv[2], sep=',', header=0)
        y_train = pd.read_csv(sys.argv[3], sep=',', header=0)
        y_test = pd.read_csv(sys.argv[4], sep=',', header=0)
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <X_train> <X_test> <y_train> <y_test>")

    model = make_pipeline(TfidfVectorizer(tokenizer=custom_tokenize), MultinomialNB())
    model.fit(X_train["Content Cleaned"], y_train["Label"])

    # Predictions on test set
    pred = model.predict(X_test["Content Cleaned"])
    np.savetxt('./output/y_imbalanced_NB_propertokens.csv', pred, delimiter=',')

    evaluate_classifier(y_test["Label"], pred, "./output/Imbalanced_NB_propertokens.png")

    return None


if __name__ == "__main__":
    main()
