import pandas as pd
import numpy as np
import sys
import json
import logging
import argparse
from utils import (
    assert_column,
    load_dtj,
    setup_logger,
    substring_check
)


debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Calculate the document-terms matrix from a document-terms JSON file.')
    parser.add_argument('infile', action="store", type=str,
                        help="input JSON file with terms vs document association")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    # TD-IDF not yet implemented
    #parser.add_argument('--type', '-t', metavar='TYPE',
    #        help='matrix elemets: TD-IDF or plain count (default)')
    return parser


def dtj_to_dtd(dtj):
    dtd = {}   # document-terms dict
    for item in dtj:
        term = item['term']
        if term not in dtd:
            dtd[term] = {}
        for o in item['occurrences']:
            dtd[term][o] = len(item['occurrences'][o])
    return dtd


def dtd_to_df(dtd):
    df = pd.DataFrame.from_dict(dtd, orient='index')
    df.fillna(0, inplace=True)
    headers = list(df.columns.values)
    headers.sort(key=lambda r:int(r))
    df = df.reindex(columns=headers)
    return df


def main():
    parser = init_argparser()
    args = parser.parse_args()

    dtj = load_dtj(args.infile)
    #dtj = dtj[1000:1050]
    #print(dtj)

    dtd = dtj_to_dtd(dtj)
    df = dtd_to_df(dtd)
    #print(df)

    # write to output, either a file or stdout (default)
    df = df.astype(int)
    output_file = open(args.output, 'w',
                       encoding='utf-8') if args.output is not None else sys.stdout
    export_csv = df.to_csv(output_file, header=True, sep='\t')
    output_file.close()


if __name__ == "__main__":
    main()

