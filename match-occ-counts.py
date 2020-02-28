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
    nrel = pandas.Series(zip(nrelevant['Count'], nrelevant['Terms']),
                         index=nrelevant['Abstract'])

    if args.output is not None:
        output = open(args.output, 'w', encoding='utf-8')
    else:
        output = sys.stdout

    header = ['abstract', 'keyword_count', 'not_relevant_count',
              'keyword', 'not-relevant']
    print('\t'.join(header), file=output)
    tot = len(keywords['Abstract'])
    for ik, k in keywords.iterrows():
        abst = k['Abstract']
        if abst in nrel:
            data = [str(abst), str(k['Count']), str(nrel[abst][0]), k['Terms'],
                    nrel[abst][1]]
            print('\t'.join(data), file=output)

        if (ik % 50) == 0:
            debug_log.debug(f'Matching keywords and not-relevant {ik}/{tot}')


if __name__ == "__main__":
    main()
