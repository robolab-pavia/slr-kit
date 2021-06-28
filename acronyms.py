import csv
import sys
import logging
import pandas as pd

import arguments
from schwartz_hearst import extract_abbreviation_definition_pairs
from utils import setup_logger


def init_argparser():
    """Initialize the command line parser."""
    parser = arguments.ArgParse()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name',
                        suggest_suffix='_acronyms.csv')
    parser.add_argument('--column', '-c', default='abstract',
                        help='Name of the column of datafile to search the '
                             'acronyms. Default: %(default)s')
    parser.add_argument('--logfile', default='slr-kit.log',
                        help='log file name. If omitted %(default)r is used',
                        logfile=True)
    return parser


def extract_acronyms(dataset, column):
    acro = {}
    for abstract in dataset[column]:
        pairs = extract_abbreviation_definition_pairs(doc_text=abstract)
        acro.update(pairs)

    acrolist = [f'{acro[abb]} | ({abb})' for abb in sorted(acro.keys())]
    return acrolist


def acronyms(args):
    debug_logger = setup_logger('debug_logger', args.logfile,
                                level=logging.DEBUG)

    # na_filter=False avoids NaN if the abstract is missing
    dataset = pd.read_csv(args.datafile, delimiter='\t', na_filter=False,
                          encoding='utf-8')
    acrolist = extract_acronyms(dataset, args.column)

    if args.output is not None:
        output_file = open(args.output, 'w', encoding='utf-8')
    else:
        output_file = sys.stdout

    writer = csv.writer(output_file, delimiter='\t', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'term', 'label'])
    for i, acro in enumerate(acrolist):
        writer.writerow([i, acro, ''])


def main():
    parser = init_argparser()
    args = parser.parse_args()
    acronyms(args)


if __name__ == "__main__":
    main()
