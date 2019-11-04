import pandas
# Libraries for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import re
import sys
import json
import logging
import argparse


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """Function to setup a generic loggers."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('abstracts', action="store", type=str,
                        help="input CSV data file with abstracts")
    parser.add_argument('terms', action="store", type=str,
                        help="input CSV data file with terms")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in line-wise JSON format')
    parser.add_argument('--label', '-l', metavar='LABEL',
                        help='label to consider for the processing')
    return parser


def main():
    abstracts_column = 'abstract_lem'
    terms_column = 'keyword'
    parser = init_argparser()
    args = parser.parse_args()

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)
    # TODO: write log string with values of the parameters used in the execution

    # load the datasets
    abstracts = pandas.read_csv(args.abstracts, delimiter='\t')
    abstracts.fillna('', inplace=True)
    if abstracts_column not in abstracts:
        print('File "{}" must contain a column labelled as "{}".'.format(args.abstracts, abstracts_column))
        sys.exit(1)
    terms = pandas.read_csv(args.terms, delimiter=',')
    terms.fillna('', inplace=True)
    if terms_column not in terms:
        print('File "{}" must contain a column labelled as "{}".'.format(args.terms, terms_column))
        sys.exit(1)
    #logging.debug(dataset.head())

    # select only the terms that are properly labelled
    if args.label is not None:
        label = args.label
    else:
        label = 'keyword'   # default behavior
    #print(label)
    keyword_flags = terms['label'] == label
    keywords = terms[keyword_flags]
    #print(keywords)

    # TODO: encapsulate the processing into a callable function

    output_file = open(args.output, 'w') if args.output is not None else sys.stdout

    # loop over all the selected terms
    tot = len(keywords[terms_column])
    for i, k in enumerate(keywords[terms_column]):
        info = {'term': k, 'occurrences': {}}
        # prepend and append a space to both keyword and abstract
        # in order to look for terms delimited by spaces (whole words)
        k_expanded = ' ' + k + ' '
        for index, row in abstracts.iterrows():
            l = []
            abs_expanded = ' ' + row[abstracts_column] + ' '
            for match in re.finditer(k_expanded, abs_expanded):
                l.append(match.start())
                #l.append((match.start(), match.end()))
            if len(l) > 0:
                info['occurrences'][row['id']] = l
        output_file.write(json.dumps(info))
        output_file.write('\n')
        if (i % 5) == 0:
            debug_logger.debug('Finding occurrences ({}/{})'.format(i, tot))
    output_file.close()


if __name__ == "__main__":
    main()
