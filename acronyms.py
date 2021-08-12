import csv
import sys
import logging
import pandas as pd

from slrkit_utils.argument_parser import ArgParse
from schwartz_hearst import extract_abbreviation_definition_pairs
from utils import setup_logger, assert_column


def to_record(config):
    """
    Returns the list of files to record with git
    :param config: content of the script config file
    :type config: dict[str, Any]
    :return: the list of files
    :rtype: list[str]
    :raise ValueError: if the config file does not contains the right values
    """
    out = config['output']
    if out is None or out == '':
        return []

    return [str(out)]


def init_argparser():
    """Initialize the command line parser."""
    parser = ArgParse()
    parser.add_argument('datafile', action="store", type=str,
                        help='input CSV data file', input=True)
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name',
                        suggest_suffix='_acronyms.csv', output=True)
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
    try:
        dataset = pd.read_csv(args.datafile, delimiter='\t', na_filter=False,
                              encoding='utf-8')
    except FileNotFoundError:
        msg = 'Error: datafile {!r} not found'
        sys.exit(msg.format(args.datafile))

    assert_column(args.datafile, dataset, args.column)

    # filter the paper using the information from the filter_paper script
    try:
        dataset = dataset[dataset['status'] == 'good'].copy()
    except KeyError:
        # no column 'status', so no filtering
        pass
    else:
        dataset.drop(columns='status', inplace=True)
        dataset.reset_index(drop=True, inplace=True)

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
