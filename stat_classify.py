from sklearn.linear_model import LogisticRegression
import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler, SMOTE
from evaluation import evaluate_classifier
from encode import tfidf_encode, count_encode, doc_2_vec, vector_for_learning, create_tagged_docs


def main():
    try:
        encode = sys.argv[1]
        train = pd.read_csv(sys.argv[2], sep=',', header=0)
        test = pd.read_csv(sys.argv[3], sep=',', header=0)
        balance = sys.argv[4]
        classifier = sys.argv[5]

    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <encoding_method> <train_file> <test_file> <balance> <classifier>")

    # Building the model

    # encoding
    if encode == "tfidf":
        vec = tfidf_encode(train)
        X_encoded = vec.transform(train["Content Cleaned"])
        X_test = vec.transform(test["Content Cleaned"])

    elif encode == "doc2vec":

        X_encoded, X_test = doc_2_vec(train, test)
        print("doc 2 vec done")

    else:
        vec = count_encode(train)
        X_encoded = vec.transform(train["Content Cleaned"])
        X_test = vec.transform(test["Content Cleaned"])

    if classifier == "LogReg":
        # balancing the data set
        if balance == "rand_ov_samp":
            X, y = RandomOverSampler(random_state=0).fit_resample(X_encoded, train["Label"])
            classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=200)

        elif balance == "smote":

            X, y = SMOTE().fit_resample(X_encoded, train["Label"])

            (unique, counts) = np.unique(y, return_counts=True)
            print('label, Count of labels')
            print(np.asarray((unique, counts)).T)

            classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=200)

        else:  # balance == "class_weight"
            classifier = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=200,
                                        class_weight='balanced')
            X = X_encoded
            y = train["Label"]

    else:
        if balance == "rand_ov_samp":
            X, y = RandomOverSampler(random_state=0).fit_resample(X_encoded, train["Label"])
            classifier = MultinomialNB()

        elif balance == "smote":
            X, y = SMOTE().fit_resample(X_encoded, train["Label"])
            (unique, counts) = np.unique(y, return_counts=True)
            print('label, Count of labels')
            print(np.asarray((unique, counts)).T)
            classifier = MultinomialNB()

        else:  # balance == "class_weight"
            classifier = MultinomialNB()
            X = X_encoded
            y = train["Label"]

    classifier.fit(X, y)

    pred = classifier.predict(X_test)

    np.savetxt(f'./output/y_{classifier}_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/{classifier}_{encode}_{balance}.png")

    return None


if __name__ == "__main__":
    main()