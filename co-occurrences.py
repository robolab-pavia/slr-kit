import pandas

import re
import sys
import json
import logging
import argparse
from utils import setup_logger


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

    with open(args.occurrence) as input_file:
        terms = []
        for l in input_file:
            terms.append(json.loads(l))

    #terms = terms[-100:]
    #for t in terms:
    #    print(t)
    #return

    co_occ = {}
    tot = len(terms) ** 2
    done = 0
    for i, t1 in enumerate(terms[:-1]):
        for t2 in terms[i + 1:]:
            if t1 == t2:
                continue
            for abs_id in t1['occurrences']:
                if abs_id in t2['occurrences']:
                    tup1 = (t1['term'], t2['term'])
                    tup2 = (t2['term'], t1['term'])
                    if tup1 in co_occ:
                        co_occ[tup1] += 1
                    else:
                        co_occ[tup1] = 1
            done += 1
            if done % 100000 == 0:
                debug_logger.debug('co-occurrences {}/{} ({:.2f})%'.format(done, tot, done / tot * 100))
    for occ in co_occ:
        print('{}\t{}\t{}'.format(occ[0], occ[1], co_occ[occ]))
    return

                        
    df = pandas.DataFrame(l, columns=['Abstract', 'Count', 'Terms'])
    df = df.sort_values(by=['Count'])

    # write to output, either a file or stdout (default)
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = df.to_csv(output_file, index=None, header=True, sep='\t')


if __name__ == "__main__":
    main()
