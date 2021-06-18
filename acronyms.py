import sys
import logging
import pandas

import arguments
from schwartz_hearst import extract_abbreviation_definition_pairs
from utils import setup_logger


def init_argparser():
    """Initialize the command line parser."""
    parser = arguments.ArgParse()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name')
    return parser


def extract_acronyms(dataset):
    acro = {}
    for abstract in dataset['abstract']:
        #print(abstract)
        pairs = extract_abbreviation_definition_pairs(doc_text=abstract)
        acro.update(pairs)
    acrodf = pandas.DataFrame(list(acro.items()), columns=['Acronym', 'Extended'])
    return acrodf


def main():
    parser = init_argparser()
    args = parser.parse_args()

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    # na_filter=False avoids NaN if the abstract is missing
    dataset = pandas.read_csv(args.datafile, delimiter='\t', na_filter=False,
                              encoding='utf-8')
    acrodf = extract_acronyms(dataset)
    acrodf = acrodf.sort_values(by=['Acronym'])
    #print(acrodf)
    output_file = open(args.output, 'w',
                       encoding='utf-8') if args.output is not None else sys.stdout
    export_csv = acrodf.to_csv(output_file, index=None, header=True, sep='\t')


if __name__ == "__main__":
    main()
