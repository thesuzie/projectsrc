import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import multiprocessing
from sklearn import utils


## ENCODING FUNCTIONS FOR BASELINE CLASSIFIERS ##

def main():

    encode = "count"

    train = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/train_IM_no8.csv")
    test = pd.read_csv("/Users/suziewelby/year3/compsci/project/src/test_train/test_IM_no8.csv")

    if encode == "tfidf":
        vec = tfidf_encode(train)
        X_encoded = vec.transform(train["Content Cleaned"])
        X_test = vec.transform(test["Content Cleaned"])

    elif encode == "doc2vec":

        X_encoded, X_test = doc_2_vec(train, test)


    else:
        vec = count_encode(train)
        X_encoded = vec.transform(train["Content Cleaned"])
        X_test = vec.transform(test["Content Cleaned"])

    train["Encoded"] = X_encoded
    test["Encoded"] = X_test

    train.to_csv(f"/Users/suziewelby/year3/compsci/project/src/test_train/{encode}_train_IM.csv")
    test.to_csv(f"/Users/suziewelby/year3/compsci/project/src/test_train/{encode}_test_IM.csv")

    return None


def custom_tokenize(content):
    # content is a string

    # tokenize
    tokens = content.split()

    # lemmatize - can try with and without this
    ##lemmatizer = WordNetLemmatizer()
    #lemmas = [lemmatizer.lemmatize(t) for t in tokens]

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
    # INPUT: txt is a dataframe
    # OUTPUT: tfidf object
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf = vectorizer.fit(txt["Content Cleaned"])

    #df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    # print(df)

    return tfidf


def count_encode(txt):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    count = vectorizer.fit(txt["Content Cleaned"])

    return count


## doc2vec encoding ##


def create_tagged_docs(txt):
    tagged = []
    stops = set(stopwords.words("english"))
    i=0

    while i < len(txt["Content Cleaned"]):
        tokens = word_tokenize(txt["Content Cleaned"][i])
        tokens = [t for t in tokens if i not in stops]
        tagged.append(TaggedDocument(words=tokens, tags=[txt["Label"][i]]))
        i+=1
    print("tagged docs")

    return tagged


def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    print("vectors made")
    return targets, feature_vectors


def doc_2_vec(train, test):
    # input train, test
    # output X_encoded, X_test

    train_documents = create_tagged_docs(train)
    test_documents = create_tagged_docs(test)

    try:
        model_dbow = Doc2Vec.load("/Users/suziewelby/year3/compsci/project/src/trained.d2v")

    except FileNotFoundError:
        cores = multiprocessing.cpu_count()

        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.025,
                             min_alpha=0.001)
        model_dbow.build_vocab([x for x in tqdm(train_documents)])
        train_documents = utils.shuffle(train_documents)
        model_dbow.train(train_documents, total_examples=len(train_documents), epochs=30)
        model_dbow.save("/Users/suziewelby/year3/compsci/project/src/trained.d2v")

    y, X_encoded = vector_for_learning(model_dbow,train_documents)
    y_test, X_test = vector_for_learning(model_dbow, test_documents)


    return X_encoded, X_test


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
    for content in train["Content Cleaned"]:
        for token in nn_tokenize(content):
            token_vocab.add(token)

    vocab = list(token_vocab)
    size = len(vocab)
    print(f"Size of Vocab: {size}")
    return vocab


def token_index(tok, vocab):
    oov = len(vocab) + 1 #shifted so that vocab indicies is 1 to max
    ind = tok

    if tok in vocab:  # if token in vocabulary
        ind = vocab.index(tok) + 1
    else:  # else it's OOV
        ind = oov
    return ind


def indices(content, vocab):
    tokens = word_tokenize(content)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stop_words]
    tok_inds = [token_index(t, vocab) for t in filtered_sentence]
    return tok_inds


def nn_encode(train, test):
    vocab = create_token_vocab(train)
    train_inds = [indices(content, vocab) for content in train["Content Cleaned"]]

    test_inds = [indices(content, vocab) for content in test["Content Cleaned"]]

    return train_inds, test_inds


if __name__ == "__main__":
    main()