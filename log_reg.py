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

    """
    try:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <train_file> <test_file>")
    """
    train_file = "/Users/suziewelby/year3/compsci/project/src/data/data0annotated.csv"
    test_file = "/Users/suziewelby/year3/compsci/project/src/data/data0annotated.csv"

    train = pd.read_csv(train_file, header=0, sep=',')
    encode_train = tfidf_encode(train[0:70])

    test = pd.read_csv(test_file,header=0, sep=',')
    encode_test = tfidf_encode(test[70:100])

    context_categories = [1,2,3,4,5,6,7,8,9]
    # print(context_categories)

    x_train = encode_train.toarray()[0:70]
    y_train = train['Label'][0:70]

    x_test = encode_test.toarray()[70:100]
    y_test = test['Label'][70:100]

    # print(f"x_train: {x_train}")
    # print(f"x_test: {x_test}")
    # print(f"y_test: {y_train}")

    # Building the model
    classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(x_train, y_train)
    print("classifier made")
    # Predictions on test set
    preds = classifier.predict(x_test)

    evaluate_classifier(y_test, preds)



if __name__ == "__main__":
    main()