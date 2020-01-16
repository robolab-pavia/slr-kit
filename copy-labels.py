import pandas
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
    parser = argparse.ArgumentParser(description='Copy labels from a CSV with labelled terms to an existing CSV without labels.')
    parser.add_argument('source', action="store", type=str,
                        help="source CSV data file with terms")
    parser.add_argument('dest', action="store", type=str,
                        help="destination CSV data file with terms")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='CSV file with the list of terms appearing in the documents')
    #parser.add_argument('--overwrite', '-w',
    #                    help='overwrite existing labels in the destination file')
    #parser.add_argument('--label', '-l', metavar='LABEL',
    #                    help='label to consider for the processing')
    return parser


def save_to_file_or_stdout(filename, df):
    output_file = open(filename, 'w') if filename is not None else sys.stdout
    export_csv = df.to_csv(output_file, header=True, sep='\t', index=False)
    output_file.close()


def copy_labels(dest_df, source_df):
    """
    Copy the labels from the terms in the source_df to the corresponding
    terms in the dest_df.
    """
    # convert the dataframe into a dictionary for faster lookup
    source_dict = {}
    for item in source_df.itertuples():
        source_dict[item.keyword] = item.label
    # iterate through the dest dataframe to update its labels
    result_df = dest_df
    for i, result_row in result_df.iterrows():
        keyword = result_row['keyword']
        if keyword in source_dict:
            result_df.at[i, 'label'] = source_dict[keyword]
    return result_df


def main():
    parser = init_argparser()
    args = parser.parse_args()

    source_df = load_df(args.source, required_columns=['keyword', 'label'])
    dest_df = load_df(args.dest, required_columns=['keyword', 'label'])
    result_df = copy_labels(dest_df, source_df)
    save_to_file_or_stdout(args.output, result_df)


if __name__ == "__main__":
    main()
