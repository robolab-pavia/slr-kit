import argparse
import sys
import logging
import pandas
from schwartz_hearst import extract_abbreviation_definition_pairs


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """Function to setup a generic loggers."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name')
    return parser


def extract_acronyms(dataset):
    acro = {}
    for abstract in dataset['abstract1']:
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

    dataset = pandas.read_csv(args.datafile, delimiter='\t')
    acrodf = extract_acronyms(dataset)
    acrodf = acrodf.sort_values(by=['Acronym'])
    #print(acrodf)
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = acrodf.to_csv(output_file, index=None, header=True, sep='\t')


if __name__ == "__main__":
    main()
