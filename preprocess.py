import pandas as pd
# Libraries for text preprocessing
import re
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

import sys
import logging
import argparse
from utils import (
    setup_logger,
    assert_column
)

BARRIER_PLACEHOLDER = 'XXX'


class AppendMultipleFilesAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            if ((isinstance(nargs, str) and nargs in ['*', '?'])
                    or (isinstance(nargs, int) and nargs < 0)):
                raise ValueError(f'nargs = {nargs} is not allowed')

        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        files = getattr(namespace, self.dest, None)
        if files is None:
            files = set()

        if not isinstance(values, list):
            values = [values]

        for v in values:
            files.add(v)

        setattr(namespace, self.dest, files)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action='store', type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME', default='-',
                        help='output file name. If omitted or %(default)s '
                             'stdout is used')
    parser.add_argument('--stop-words', '-s', action=AppendMultipleFilesAction,
                        nargs='+', metavar='FILENAME', dest='stop_words_file',
                        help='stop words file name')
    return parser


def load_stop_words(input_file, language='english'):
    with open(input_file, 'r', encoding='utf-8') as f:
        stop_words_list = f.read().splitlines()

    stop_words_list = [w for w in stop_words_list if w != '' and w[0] != '#']
    # Creating a list of custom stopwords
    return set(stop_words_list)


def preprocess_item(item, barrier_words):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', item)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub('&lt;/?.*?&gt;', ' &lt;&gt; ', text)
    # Remove special characters and digits
    text = re.sub('(\\d|\\W)+', ' ', text)
    # Convert to list from string
    text = text.split()
    # Lemmatisation
    lem = WordNetLemmatizer()
    # text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text2 = []
    for word in text:
        text2.append(lem.lemmatize(word))

    # TODO: mark relevant words

    for i, word in enumerate(text2):
        if word in barrier_words:
            text2[i] = BARRIER_PLACEHOLDER

    return text2


def process_corpus(dataset, stop_words):
    corpus = []
    for item in dataset:
        text = preprocess_item(item, stop_words)
        text = ' '.join(text)
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
    dataset = pd.read_csv(args.datafile, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    assert_column(args.datafile, dataset, target_column)
    debug_logger.debug('Dataset loaded {} items'.format(len(dataset[target_column])))

    stop_words = set()

    if args.stop_words_file is not None:
        for sfile in args.stop_words_file:
            stop_words |= load_stop_words(sfile)

        debug_logger.debug('Stopwords loaded and updated')

    corpus = process_corpus(dataset[target_column], stop_words)
    debug_logger.debug('Corpus processed')
    dataset['abstract_lem'] = corpus

    # write to output, either a file or stdout (default)
    if args.output == '-':
        output_file = sys.stdout
    else:
        output_file = open(args.output, 'w', encoding='utf-8')

    dataset.to_csv(output_file, index=None, header=True, sep='\t')


if __name__ == '__main__':
    main()
