import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



# TODO:
# encoding

# need to have just one main function that takes in the dataframe and returns it encoded
# could do one main for tokenisation and cleaning and then each new one for different encodings


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
    ignore = "(Start contract)|(Code [.*])|(Spoiler \(Click to view\()"
    cleaned = re.sub(ignore, "", cleaned)

    # remove punctuation and numbers
    cleaned = re.sub('[^a-zA-Z]+', " ", cleaned)

    # make lower case
    clean_content = cleaned.lower()

    # tokenize
    tokens = word_tokenize(clean_content)

    # remove stopwords
    remove_sw = [w for w in tokens if not w in stopwords.words()]
    print(remove_sw)

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


data = pd.read_csv('/Users/suziewelby/year3/compsci/project/src/data/data0annotated.csv', header=0, sep=',')
#with_tokens = create_tokens(data)
vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)
tfidf = vectorizer.fit_transform(data['Content Cleaned'][0:10])

df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
print(df)

#with_tokens.to_csv('/Users/suziewelby/year3/compsci/project/src/data/data0tokens.csv')
