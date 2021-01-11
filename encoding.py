import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# TODO:
# encoding

# need to have just one main function that takes in the dataframe and returns it encoded
# could do one main for tokenisation and cleaning and then each new one for different encodings

def clean(content):  # content is a string
    # remove links
    links = "(http\S*)|(\S*.co\S*)|(\S*.net\S*)"
    cleaned = re.sub(links, "", content)

    # remove usernames and emails (strings containing @)
    usrnm = "\S*@\S*"
    cleaned = re.sub(usrnm, "", cleaned)

    # remove punctuation and numbers
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    cleaned = cleaned.translate(str.maketrans('', '', string.digits))

    # make lower case
    cleaned.lower()

    return cleaned


def tokenize(clean_content):  # content is a string

    # tokenize
    tokens = word_tokenize(clean_content)

    # remove stopwords
    remove_sw = [w for w in tokens if w not in stopwords.words()]

    return remove_sw


def create_tokens(txt): # input is dataframe
    tokens = [tokenize(clean(t)) for t in txt['Content Cleaned']]
    txt['Tokens'] = tokens
    return txt


data = pd.read_csv('./data/data0annotated.csv', header=0, sep=',')
with_tokens = create_tokens(data)
with_tokens.to_csv('./data/data0tokens.csv')

