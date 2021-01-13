import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score



# TODO:
# split into different sets
# naive bayes implementation
# evaluation functions and graphs/confusion matrices


def clean(content):  # content is a string
    # now moved into the whole tokenise function
    # remove links
    links = "(http\S*)|(\S*.co\S*)|(\S*.net\S*)"
    cleaned = re.sub(links, "", content)

    # remove usernames and emails (strings containing @)
    usrnm = "\S*@\S*"
    cleaned = re.sub(usrnm, "", cleaned)

    # remove punctuation and numbers
    punct = string.punctuation
    punct.__add__('’')
    cleaned = cleaned.translate(str.maketrans('', '', punct))
    cleaned = cleaned.translate(str.maketrans('', '', string.digits))

    # make lower case
    lower_cleaned = cleaned.lower()

    return lower_cleaned


def custom_tokenize(content):  # content is a string
    # remove links
    links = "(http\S*)|(\S*\.co\S*)|(\S*\.net\S*)"
    cleaned = re.sub(links, "", content)

    # remove usernames and emails (strings containing @)
    usrnm = "\S*@\S*"
    cleaned = re.sub(usrnm, "", cleaned)

    # remove unneeded indicator words that have been added
    ignore = "(Start contract)|(Code [.*])|(Spoiler \(Click to View\))"
    cleaned = re.sub(ignore, "", cleaned)

    # remove punctuation and numbers
    cleaned = re.sub('[^a-zA-Z]+', " ", cleaned)

    # make lower case
    clean_content = cleaned.lower()

    # tokenize
    tokens = word_tokenize(clean_content)

    # remove stopwords
    remove_sw = [w for w in tokens if not w in stopwords.words()]

    return remove_sw


def create_tokens(txt):  # input is dataframe
    # not actually needed - the tfidf_encode will do this on its own, this would only be used for illustration
    # clean_content = [clean(t) for t in txt['Content Cleaned']]
    tokens = [custom_tokenize(t) for t in txt['Content Cleaned']]
    txt['Tokens'] = tokens
    return txt


def tfidf_encode(txt):
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)
    tfidf = vectorizer.fit_transform(txt['Content Cleaned'])

    return tfidf


def main():

    try:
        file = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <data_file>")

    data = pd.read_csv(file, header=0, sep=',')

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)

    tfidf = vectorizer.fit_transform(data['Content Cleaned'][0:100])

    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    #print(df)

    context_categories  = np.unique(data['Label'][0:100])
    #print(context_categories)

    # splitting into testing a training for now!
    # todo: also make validation set
    # todo: make my own function for this - or look into options more to have stratified sampling
    x_train, x_test, y_train, y_test = train_test_split(tfidf.toarray(), data['Label'][0:100])
    #print(f"x_train: {x_train}")
    #print(f"x_test: {x_test}")

    #print(f"y_test: {y_train}")

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
