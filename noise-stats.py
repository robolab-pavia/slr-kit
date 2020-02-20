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
    parser = argparse.ArgumentParser()
    parser.add_argument('terms', action="store", type=str,
                        help="input CSV data file with terms")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    parser.add_argument('--label', '-l', metavar='LABEL',
                        help='label to consider for the processing (default is "noise"')
    return parser


def main():
    parser = init_argparser()
    args = parser.parse_args()

    # load the dataset
    df_terms = pd.read_csv(args.terms, delimiter='\t', encoding='utf-8')
    df_terms.fillna('', inplace=True)
    assert_column(args.terms, df_terms, 'keyword')
    assert_column(args.terms, df_terms, 'label')
    debug_logger.debug(df_terms.head())

    # select only the terms that are properly labelled
    label = 'noise'   # default behavior
    if args.label is not None:
        label = args.label

    terms = {}
    for i, row in df_terms.iterrows():
        tokens = row['keyword'].split()
        for t in tokens:
            if t not in terms:
                terms[t] = {'Total': 1, label: 0, 'Not Labelled': 0}
            else:
                terms[t]['Total'] += 1
            if row['label'] == label:
                terms[t][label] += 1
            elif row['label'] == '':
                terms[t]['Not Labelled'] += 1
    #print(terms)

    # this is the desired ordering of the columns
    headers = [
        'Index',
        'Total',
        label,
        'Not Labelled'
    ]
    # add the calculated index to the dictionary
    for t in terms:
        total = terms[t]['Total']
        index = (terms[t][label] + terms[t]['Not Labelled']) / total
        terms[t]['Index'] = index

    df = pd.DataFrame.from_dict(terms, orient='index')
    # reordering of the columns
    df = df[headers]
    sorted_df = df.sort_values(['Index', 'Total', label], axis=0, ascending=False, kind='quicksort')
    #print(df)

    # write to output, either a file or stdout (default)
    output_file = open(args.output, 'w',
                       encoding='utf-8') if args.output is not None else sys.stdout
    export_csv = sorted_df.to_csv(output_file, header=True, sep='\t')

if __name__ == "__main__":
    main()

