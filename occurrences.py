import pandas
import re
import sys
import json
import logging
import argparse
from multiprocessing import Pool
from utils import (
    setup_logger,
    load_df
)


debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

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


def find_all_occurrences(term, document, doc_id):
    l = []
    # prepend and append a space to both keyword and abstract
    # in order to look for terms delimited by spaces (whole words)
    term_expanded = ' ' + term + ' '
    doc_expanded = ' ' + document + ' '
    for match in re.finditer(term_expanded, doc_expanded):
        l.append(match.start())
    return doc_id, l


def find_occurrences_in_all_documents(terms, documents_df, output_file):
    # loop over all the terms
    tot = len(terms)
    document_list = list(documents_df['abstract_lem'])
    doc_id_list = list(documents_df['id'])
    for i, term in enumerate(terms):
        info = {'term': term, 'occurrences': {}}
        term_list = [term] * len(document_list)
        # parallel invocation of the search function
        with Pool() as pool:
            result = pool.starmap(find_all_occurrences, zip(term_list, document_list, doc_id_list))
        # keeps non empty lists of occurrences
        info['occurrences'] = {r[0]: r[1] for r in result if len(r[1]) > 0}
        output_file.write(json.dumps(info))
        output_file.write('\n')
        if (i % 5) == 0:
            debug_logger.debug('Finding occurrences ({}/{})'.format(i, tot))


def main():
    abstracts_column = 'abstract_lem'
    terms_column = 'keyword'
    parser = init_argparser()
    args = parser.parse_args()

    # TODO: write log string with values of the parameters used in the execution

    # load the datasets
    abstracts_df = load_df(args.abstracts, required_columns=[abstracts_column])
    terms_df = load_df(args.terms, required_columns=[terms_column])
    #logging.debug(dataset.head())

    # select only the terms that are properly labelled
    if args.label is not None:
        label = args.label
    else:
        label = 'keyword'   # default behavior
    #print(label)
    terms_flags = terms_df['label'] == label
    terms_df = terms_df[terms_flags]
    #print(keywords)

    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    find_occurrences_in_all_documents(terms_df[terms_column], abstracts_df, output_file)
    output_file.close()


if __name__ == "__main__":
    main()
