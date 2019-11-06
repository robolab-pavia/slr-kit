import pandas
import sys
import csv
import logging
import argparse


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
    parser.add_argument('--stop-words', '-s', metavar='FILENAME', dest='stop_words_file',
                        help='stop words file name')
    return parser


def main():
    #parser = init_argparser()
    #args = parser.parse_args()
    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    # load the dataset
    # TODO: read the input files from command line
    keywords = pandas.read_csv('occ-keyword-count.csv', delimiter='\t')
    keywords.fillna('', inplace=True)
    nrelevant = pandas.read_csv('occ-not-relevant-count.csv', delimiter='\t')
    nrelevant.fillna('', inplace=True)

    # convert the dataframe into a Series indexed by Abstract
    # making the search of an existing abstract very quick
    nrel = pandas.Series(zip(nrelevant['Count'], nrelevant['Terms']), index=nrelevant['Abstract'])

    print('{}\t{}\t{}\t{}\t{}'.format('Abstract', 'Keyword count', 'Not-relevant count', 'Keyword Terms', 'Not-relevant Terms'))
    tot = len(keywords['Abstract'])
    for ik, k in keywords.iterrows():
        abst = k['Abstract']
        if abst in nrel:
            print('{}\t{}\t{}\t{}\t{}'.format(abst, k['Count'], nrel[abst][0], k['Terms'], nrel[abst][1]))
        if (ik % 50) == 0:
            debug_logger.debug('Matching keywords and not-relevant {}/{}'.format(ik, tot))
    

if __name__ == "__main__":
    main()
