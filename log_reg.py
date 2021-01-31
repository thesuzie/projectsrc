from sklearn.pipeline import make_pipeline

from encode_simple import custom_tokenize, tfidf_encode
import sys
from nltk.corpus import stopwords
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
        encode = sys.argv[1]
        train = pd.read_csv(sys.argv[2], sep=',', header=0)
        test = pd.read_csv(sys.argv[3], sep=',', header=0)
        balance = sys.argv[4]

    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <encoding_method> <train_file> <test_file> ")

    # Building the model

    model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')), LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg'))
    model.fit(train["Content Cleaned"], train["Label"])
    pred = model.predict(test["Content Cleaned"])

    np.savetxt(f'./output/y_LogReg_{encode}_{balance}.csv', pred, delimiter=',')

    evaluate_classifier(test["Label"], pred, f"./output/LogReg_{encode}_{balance}.png")


if __name__ == "__main__":
    main()