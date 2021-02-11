from encode_simple import tfidf_encode, count_encode
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_classifier


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
        vec = tfidf_encode(train)
    else:
        vec = count_encode(train)

    X_encoded = vec.transform(train["Content Cleaned"])

    if balance == "rand_ov_samp":
        X, y = RandomOverSampler(random_state=0).fit_resample(X_encoded, train["Label"])
        classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg')

    elif balance == "smote":
        X, y = SMOTE().fit_resample(X_encoded, train["Label"])
        classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg')

    else: # balance == "class_weight"
        classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg', class_weight='balanced')
        X = X_encoded
        y = train["Label"]

    classifier.fit(X, y)

    X_test = vec.transform(test["Content Cleaned"])

    pred = classifier.predict(X_test)

    np.savetxt(f'./output/y_LogReg_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/LogReg_{encode}_{balance}.png")

    return None


if __name__ == "__main__":
    main()