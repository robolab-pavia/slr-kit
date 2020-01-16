from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sys
import logging
import argparse
from utils import (
    assert_column,
    setup_logger,
    substring_check
)


debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Calculate the cosine similarity of a document-terms matrix.')
    parser.add_argument('terms', action="store", type=str,
                        help="input CSV file containing the document-terms matrix")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def save_term_list(filename, df):
    with open(filename, 'w') as outfile:
        outfile.write('id\tterm\n')
        i = 0
        for name, row in df.iterrows():
            outfile.write('{}\t{}\n'.format(i, name))
            i += 1

def main():
    debug_logger.debug('[cosine_similarity] Started')
    parser = init_argparser()
    args = parser.parse_args()

    # load the dataset
    debug_logger.debug('[cosine_similarity] Loading input file')
    df = pd.read_csv(args.terms, delimiter='\t', index_col=0)
    df.fillna('', inplace=True)
    #debug_logger.debug(df.head())

    debug_logger.debug('[cosine_similarity] Calculate similarity')
    cs = cosine_similarity(df)
    cs_pd = pd.DataFrame(cs)

    # write to output, either a file or stdout (default)
    debug_logger.debug('[cosine_similarity] Saving')
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = cs_pd.to_csv(output_file, header=True, sep='\t', float_format='%.3f')
    output_file.close()

    # TODO: allow the selection of this filename from command line
    save_term_list('term-list.csv', df)

    debug_logger.debug('[cosine_similarity] Terminated')


if __name__ == "__main__":
    main()

