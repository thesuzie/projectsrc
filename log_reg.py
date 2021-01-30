from sklearn.pipeline import make_pipeline

from encode_simple import custom_tokenize, tfidf_encode
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_classifier


def main():
    try:
        X_train = pd.read_csv(sys.argv[1], sep=',', header=0)
        X_test = pd.read_csv(sys.argv[2], sep=',', header=0)
        y_train = pd.read_csv(sys.argv[3], sep=',', header=0)
        y_test = pd.read_csv(sys.argv[4], sep=',', header=0)
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <X_train> <X_test> <y_train> <y_test>")

    # Building the model

    model = make_pipeline(TfidfVectorizer(tokenizer=custom_tokenize), LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg'))
    model.fit(X_train["Content Cleaned"], y_train["Label"])
    pred = model.predict(X_test["Content Cleaned"])

    np.savetxt('./output/y_imbalanced_LogReg_propertokens.csv', pred, delimiter=',')

    evaluate_classifier(y_test["Label"], pred, "./output/Imbalanced_LogReg_propertokens.png")


if __name__ == "__main__":
    main()