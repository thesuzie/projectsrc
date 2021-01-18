import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def custom_tokenize(content):
    # content is a string

    # remove links
    links = "(http\S*)|(\S*\.co\S*)|(\S*\.net\S*)"
    cleaned = re.sub(links, "", content)

    # remove usernames and emails (strings containing @)
    usrnm = "\S*@\S*"
    cleaned = re.sub(usrnm, "", cleaned)

    # remove unneeded indicator words that have been added
    ignore = "(Start contract)|(Code \[.*\])|(Spoiler \(Click to View\))"
    cleaned = re.sub(ignore, "", cleaned)

    # remove punctuation and numbers
    cleaned = re.sub('[^a-zA-Z]+', " ", cleaned)

    # make lower case
    clean_content = cleaned.lower()

    # tokenize
    tokens = word_tokenize(clean_content)

    # lemmatize - can try with and without this
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    # remove stopwords
    remove_sw = [w for w in lemmas if not w in stopwords.words()]

    return remove_sw


def create_tokens(txt):  # input is dataframe
    # not actually needed - the tfidf_encode will do this on its own, this would only be used for illustration
    # clean_content = [clean(t) for t in txt['Content Cleaned']]
    tokens = [custom_tokenize(t) for t in txt['Content Cleaned']]
    txt['Tokens'] = tokens
    return txt


def tfidf_encode(txt):
    # INPUT: txt is a dataframe
    # OUTPUT: tfidf object
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)
    tfidf = vectorizer.fit_transform(txt['Content Cleaned'])

    return tfidf
