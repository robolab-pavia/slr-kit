import argparse
import sys
import pandas as pd
from RISparser import readris


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', action="store", type=str,
                        help="input RIS bibliography file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output CSV file name')
    parser.add_argument('--columns', '-c', metavar='col1,..,coln',
                        help='list of comma-separated columns to export; \'?\' for the list of available columns')
    return parser


def show_columns(df):
    print('Valid columns:')
    for c in df.columns:
        print('   {}'.format(c))


def main():
    parser = init_argparser()
    args = parser.parse_args()

    output_file = open(args.output, 'w') if args.output is not None else sys.stdout

    with open(args.input_file, 'r') as bibliography_file:
        entries = readris(bibliography_file)
        risdf = pd.DataFrame(entries)

    if args.columns is not None:
        cols = args.columns.split(',')
        # checks if help was requested
        if len(cols) == 1 and cols[0] == '?':
            show_columns(risdf)
            sys.exit(1)
        # checks that the requested items exist in the RIS file
        for c in cols:
            if c not in risdf:
                print('Invalid column: "{}".'.format(c))
                sys.exit(1)
    else:
        # uses 'title' as default column
        if 'title' in risdf:
            cols = ['title']
        else:
            print('Column "title" not present; no columns specified.')
            show_columns(risdf)
            sys.exit(1)

    export_csv = risdf.to_csv(
            output_file,
            columns=cols,
            index=[i for i in risdf.index],
            index_label='id',
            header=True,
            sep='\t')


if __name__ == "__main__":
    main()
