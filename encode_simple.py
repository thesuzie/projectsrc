import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


## ENCODING FUNCTIONS FOR BASELINE CLASSIFIERS ##


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
    tfidf = vectorizer.fit_transform(txt["Content Cleaned"])

    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    # print(df)

    return tfidf


## ENCODING FUNCTIONS FOR NEURAL NETWORK ##


def nn_tokenize(content):
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
    # lemmatizer = WordNetLemmatizer()
    # lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    # remove stopwords
    # remove_sw = [w for w in lemmas if not w in stopwords.words()]

    return tokens


def create_token_vocab(train):
    token_vocab = set()
    for content in train["Content"]:
        for token in nn_tokenize(content):
            token_vocab.add(token)

    vocab = list(token_vocab)
    return vocab


def token_index(tok, vocab):
    oov = len(vocab)
    ind = tok
    if not pd.isnull(tok):  # new since last time: deal with the empty lines which we didn't drop yet
        if tok in vocab:  # if token in vocabulary
            ind = vocab.index(tok)
        else:  # else it's OOV
            ind = oov
    return ind


def indices(content, vocab):
    tokens = nn_tokenize(content)
    tokinds = [token_index(t, vocab) for t in tokens]
    return tokinds


def nn_encode(train, test):
    vocab = create_token_vocab(train)
    train["Token_indices"] = [indices(content, vocab) for content in train["Content"]]

    test["Token_indices"] = [indices(content, vocab) for content in test["Content"]]
