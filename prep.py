# Namdar Kabolinejad 2021

# This script prepares reads the CSV dataset into
# dataframes and is responsible for splitting the dataset
# into training and testing sets. It includes
# additional optional functions for further preprocessing such as
# lemmatization or stemming.

import re, nltk
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import wordnet

'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
'''

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

TRAIN_RATION = 0.8
DEV_RATIO = 0

DATA_PATH = "./CSV Lyrics/"
GENRES = ["pop", "r-b", "rap", "rock"]


def create_df():
    for i in GENRES:
        globals()["%s_data" % i] = pd.read_csv(DATA_PATH + i + ".csv", usecols=['lyrics'])


def train_validate_test_split(df, train_percent=TRAIN_RATION, validate_percent=DEV_RATIO, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def split_all():
    for i in GENRES:
        cur_df = globals()["%s_data" % i]
        _traning, _dev, _test = train_validate_test_split(cur_df)
        globals()["%s_train" % i] = _traning
        globals()["%s_dev" % i] = _dev
        globals()["%s_test" % i] = _test


def get_pos(word):
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return None


def rm_stop(words):
    split = words.split()
    rtn = []

    for word in split:
        if word not in stopword_list:
            rtn.append(word)

    rtn = " ".join(rtn)

    return rtn


def stem(words):
    rtn = []
    split = words.split()

    for word in split:
        rtn.append(PorterStemmer().stem(word))

    rtn = " ".join(rtn)
    return rtn


def lemmatize(words):
    pos_list = nltk.pos_tag(nltk.word_tokenize(words))

    rtn = []
    for word in pos_list:
        if get_pos(word[1]) is None:
            rtn.append(word[0])
        else:
            rtn.append(WordNetLemmatizer().lemmatize(word[0], get_pos(word[1])))

    rtn = " ".join(rtn)

    return rtn


def preprocess(corpus, clean=False, stop=False, lemmatization=False, stemming=False, special=False):
    normalized_corpus = []

    for index, row in corpus.iterrows():
        line = row["lyrics"]

        # clean text
        if clean:
            # lowercase the text
            line = line.lower()

            # remove extra newlines
            line = re.sub(r'[\r|\n|\r\n]+', ' ', line)

        # remove special chars
        if special:
            line = re.sub('[^a-zA-z0-9\s]', '', line)

        # remove stop words
        if stop:
            line = rm_stop(line)

        # lemmatize text
        if lemmatization:
            line = lemmatize(line)

        # stem text
        if stemming:
            line = stem(line)

        # clean text
        if clean:
            # remove extra whitespace
            line = re.sub(' +', ' ', line)

        normalized_corpus.append(line)

    return pd.DataFrame(normalized_corpus, columns=['lyrics'])


def prepare_data():
    # create the dataframes
    create_df()

    # split that data
    split_all()


def main():
    prepare_data()


main()


class Data:
    trainPop = globals()["%s_train" % "pop"]
    devPop = globals()["%s_dev" % "pop"]
    testPop = globals()["%s_test" % "pop"]
    trainRock = globals()["%s_train" % "rock"]
    devRock = globals()["%s_dev" % "rock"]
    testRock = globals()["%s_test" % "rock"]
    trainRap = globals()["%s_train" % "rap"]
    devRap = globals()["%s_dev" % "rap"]
    testRap = globals()["%s_test" % "rap"]
    trainRnB = globals()["%s_train" % "r-b"]
    devRnB = globals()["%s_dev" % "r-b"]
    testRnB = globals()["%s_test" % "r-b"]
