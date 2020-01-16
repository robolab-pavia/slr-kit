import pandas
# Libraries for text preprocessing
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

import sys
import logging
import argparse
from utils import (
    setup_logger,
    assert_column
)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name')
    parser.add_argument('--stop-words', '-s', metavar='FILENAME', dest='stop_words_file',
                        help='stop words file name')
    return parser


def load_stop_words(input_file, language='english'):
    stop_words_list = []
    with open(input_file, "r") as f:
        stop_words_list = f.read().split('\n')
    stop_words_list = [w for w in stop_words_list if w != '']
    stop_words_list = [w for w in stop_words_list if w[0] != '#']
    # Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))
    # Creating a list of custom stopwords
    new_words = stop_words_list
    stop_words = stop_words.union(new_words)
    return list(stop_words)


def preprocess_item(item, stop_words):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', item)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    # Remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ", text)
    # Convert to list from string
    text = text.split()
    # Stemming
    ps=PorterStemmer()
    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words] 
    return text


def process_corpus(dataset, stop_words):
    corpus = []
    for item in dataset:
        text = preprocess_item(item, stop_words)
        text = " ".join(text)
        corpus.append(text)
    return corpus


def main():
    target_column = 'abstract'
    parser = init_argparser()
    args = parser.parse_args()

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)
    # TODO: write log string with values of the parameters used in the execution

    # load the dataset
    dataset = pandas.read_csv(args.datafile, delimiter='\t')
    dataset.fillna('', inplace=True)
    assert_column(args.datafile, dataset, target_column)
    debug_logger.debug("Dataset loaded {} items".format(len(dataset[target_column])))
    #logging.debug(dataset.head())

    if args.stop_words_file is not None:
        stop_words = load_stop_words(args.stop_words_file, language='english')
        debug_logger.debug("Stopwords loaded and updated")
    else:
        stop_words = []
    # print(stop_words)

    corpus = process_corpus(dataset[target_column], stop_words)
    debug_logger.debug("Corpus processed")
    dataset['abstract_lem'] = corpus

    # View corpus item
    #print(corpus[0])
    #print(dataset.iloc[0])

    # write to output, either a file or stdout (default)
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = dataset.to_csv(output_file, index=None, header=True, sep='\t')


if __name__ == "__main__":
    main()
