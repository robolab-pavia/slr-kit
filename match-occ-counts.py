import pandas
import sys
import csv
import logging
import argparse
from utils import setup_logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('keyword_file', action='store', type=str,
                        help='CSV input file with the keyword occurrences')
    parser.add_argument('not_relevant_file', action='store', type=str,
                        help='CSV input file with the not-relevant occurrences')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name')
    return parser


def main():
    parser = init_argparser()
    args = parser.parse_args()
    debug_log = setup_logger('debug_logger', 'slr-kit.log', level=logging.DEBUG)

    # load the dataset
    keywords = pandas.read_csv(args.keyword_file, delimiter='\t',
                               encoding='utf-8')
    keywords.fillna('', inplace=True)
    nrelevant = pandas.read_csv(args.not_relevant_file, delimiter='\t',
                                encoding='utf-8')
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
