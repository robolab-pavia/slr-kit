import argparse
import logging
import re
import sys
from typing import Generator
from timeit import default_timer as timer

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

from utils import (
    setup_logger,
    assert_column
)

BARRIER_PLACEHOLDER = 'XXX'
RELEVANT_PREFIX = BARRIER_PLACEHOLDER


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
    parser.add_argument('--barrier-words', '-b',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='barrier_words_file',
                        help='barrier words file name')
    parser.add_argument('--relevant-term', '-r', nargs='+', metavar='FILENAME',
                        dest='relevant_terms_file',
                        action=AppendMultipleFilesAction,
                        help='relevant terms file name')
    return parser


def load_barrier_words(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        stop_words_list = f.read().splitlines()

    stop_words_list = {w for w in stop_words_list if w != '' and w[0] != '#'}
    # Creating a list of custom stopwords
    return stop_words_list


def replace_ngram(text, n_grams, check_subsequent):
    """
    Replace the given n-grams with a placeholder in the specified text

    The n-grams and their placeholder are taken from the generator n_grams, that
    must be a generator that yields a tuple (placeholder, n-gram). Each n-gram
    must be a tuple of strings.
    The function can also check if the toke immediately after the replaced
    n-gram is equal to the placeholder to remove it. This is useful for acronyms
    because, usually the abbreviation immediately follows the extended acronym.

    :param text: the text to search
    :type text: list[str]
    :param n_grams: generator that yields n-grams and their placeholder
    :type n_grams: Generator[tuple[str, tuple[str]], Any, None]
    :param check_subsequent: if True, check the token immediately after the
        found n-gram. If it is equal to the placeholder, this token is removed.
    :return: the transformed text
    :rtype: list[str]
    """
    text2 = list(text)
    for placeholder, ngram in n_grams:
        end = False
        index = -1
        length = len(ngram)
        while not end:
            try:
                # index + 1 to skip the previous match
                index = text2.index(ngram[0], index + 1)
                if tuple(text2[index:index + length]) == ngram:
                    # found!
                    text2[index:index + length] = [placeholder]
                    try:
                        if check_subsequent and text2[index + 1] == placeholder:
                            del text2[index + 1]
                    except IndexError:
                        # reached the end of the list: stop the loop
                        end = True
            except ValueError:
                end = True

    return text2


def preprocess_item(item, relevant_terms, barrier_words):
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

    # mark relevant terms
    rel_gen = ((f'{RELEVANT_PREFIX}_{"_".join(rel)}', rel)
               for rel in relevant_terms)
    text2 = replace_ngram(text2, rel_gen, False)

    for i, word in enumerate(text2):
        if word in barrier_words:
            text2[i] = BARRIER_PLACEHOLDER

    return text2


def process_corpus(dataset, relevant_terms, barrier_words):
    corpus = []
    for item in dataset:
        text = preprocess_item(item, relevant_terms, barrier_words)
        text = ' '.join(text)
        corpus.append(text)

    return corpus


def load_relevant_terms(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        rel_words_list = f.read().splitlines()

    rel_words_list = {tuple(w.split(' '))
                      for w in rel_words_list
                      if w != '' and w[0] != '#'}

    return rel_words_list


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

    barrier_words = set()

    if args.barrier_words_file is not None:
        for sfile in args.barrier_words_file:
            barrier_words |= load_barrier_words(sfile)

        debug_logger.debug('Barrier words loaded and updated')

    rel_terms = set()
    if args.relevant_terms_file is not None:
        for rfile in args.relevant_terms_file:
            rel_terms = load_relevant_terms(rfile)

        debug_logger.debug('Relevant words loaded and updated')

    start = timer()
    corpus = process_corpus(dataset[target_column], rel_terms, barrier_words)
    stop = timer()
    elapsed_time = stop - start
    debug_logger.debug('Corpus processed')
    dataset['abstract_lem'] = corpus

    # write to output, either a file or stdout (default)
    if args.output == '-':
        output_file = sys.stdout
    else:
        output_file = open(args.output, 'w', encoding='utf-8')

    dataset.to_csv(output_file, index=None, header=True, sep='\t')
    print('Elapsed time:', elapsed_time, file=sys.stderr)


if __name__ == '__main__':
    main()
