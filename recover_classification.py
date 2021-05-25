import pandas as pd
import argparse


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('old_classification', action='store', type=str,
                        help="old CSV data file partially classified")
    parser.add_argument('new_classification', action='store', type=str,
                        help="new CSV data file to be classified")
    parser.add_argument('--output', '-o', metavar='FILENAME', default='-',
                        help='output file name. If omitted or %(default)r '
                             'stdout is used')

    return parser

def main():
    parser = init_argparser()
    args = parser.parse_args()
    
    oc = pd.read_csv(args.old_classification, sep='\t',
                                        usecols=['term', 'label'])
    nc = nc1 = pd.read_csv(args.new_classification, sep='\t').dropna(subset=['term'])
    
    nc = nc.merge(oc, on='term', how='left')
    nc = nc.assign(label = nc['label_x'].combine_first(nc['label_y']))
    nc = nc.drop(['label_x', 'label_y'], axis=1)
    nc.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()