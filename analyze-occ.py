import pandas

import re
import sys
import json
import logging
import argparse
from utils import (
    setup_logger,
    load_dtj
)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('occurrence', action="store", type=str,
                        help="input JSON data file with occurrences")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='CSV file with the list of terms appearing in the documents')
    return parser


def main():
    parser = init_argparser()
    args = parser.parse_args()

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    terms = load_dtj(args.occurrence)

    occ_x_abs = {}
    for t in terms:
        for o in t['occurrences']:
            if o not in occ_x_abs:
                occ_x_abs[o] = {'count': 0, 'terms': set()}
            occ_x_abs[o]['count'] += 1
            occ_x_abs[o]['terms'].update([t['term']])
    #print(occ_x_abs)
    l = []
    for abstract in occ_x_abs:
        terms = ','.join(occ_x_abs[abstract]['terms'])
        item = [abstract, occ_x_abs[abstract]['count'], terms]
        l.append(item)
    df = pandas.DataFrame(l, columns=['Abstract', 'Count', 'Terms'])
    df = df.sort_values(by=['Count'])

    # write to output, either a file or stdout (default)
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = df.to_csv(output_file, index=None, header=True, sep='\t',
                           encoding='utf-8')


if __name__ == "__main__":
    main()
