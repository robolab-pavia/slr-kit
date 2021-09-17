"""
    This script is used to recover a classification already done.
    Let's say you had a .csv output from gen-ngrams that was already classified
    (at least partially) with FAWOC. Than you went back on preprocess or
    something like that to refine the grams to extract and classify.
    With this script you can match the new terms with no label with the old
    classifications eventually made. 
"""
import pandas as pd
import argparse


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('old', action='store', type=str,
                        help='old CSV data file partially classified')
    parser.add_argument('new', action='store', type=str,
                        help='new CSV data file to be classified')
    parser.add_argument('output', help='output file name')

    return parser


def main():
    parser = init_argparser()
    args = parser.parse_args()

    oc = pd.read_csv(args.old, sep='\t',
                     usecols=['term', 'label']).dropna(subset=['label'])

    nc = pd.read_csv(args.new, sep='\t')
    nc = nc.merge(oc, on='term', how='left')
    nc = nc.assign(label=nc['label_x'].combine_first(nc['label_y']))
    nc = nc.drop(['label_x', 'label_y'], axis=1)
    nc.to_csv(args.output, index=False, sep='\t')


if __name__ == '__main__':
    main()
