import argparse
import json
import logging
import re
from collections import defaultdict
from pprint import pprint

from utils import setup_logger

debug_logger = setup_logger('debug_logger', 'slr-kit.log', level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(
        description='Convert the output of manual clustering process to a computer-friendly JSON file')

    parser.add_argument('pairings', metavar='FILENAME', action="store", type=str,
                        help="input file in format (label:id1, id2, id3 ...)")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in JSON format')
    return parser


def parse_file(filename):
    """
    Parse the input file with manual labelled clusters, where documents can belong to multiple clusters.
    This expects as format for each line a cluster label and the list of document ID.
        --> 'real time systems: 2,5,45,1345,2456,[..]'
    :param filename: Input file name path
    :return: Dictionary with documents IDs as key and an array with cluster labels ({ '67': ['label1', 'label3']})
    """

    # line structure: label:id1,id2,id3,...
    voc = defaultdict(list)
    pattern = re.compile("(\w *)*:(\d+)(,\s*\d+)*")

    try:
        f = open(filename, 'r', encoding='utf-8')

        lines = f.readlines()

        for line in lines:

            if pattern.match(line.lstrip()):
                tokens = line.lstrip().split(':')

                label = tokens[0]
                id_list = tokens[1].split(',')

                for doc in id_list:
                    voc[int(doc)].append(label)
    except OSError:
        print("Could not open/read file:", filename)
        exit(-1)

    return voc


def dict2file(voc, out_file):
    """
    Save sorted dictionary (by ID) to JSON file as specified by argument
    :param voc:
    :param out_file:
    :return:
    """
    with open(out_file, 'w') as fp:
        json.dump(voc, fp, sort_keys=True, indent=4)


def main():

    parser = init_argparser()
    args = parser.parse_args()

    debug_logger.debug('[Parse pairings] Started')
    voc = parse_file(args.pairings)

    if args.output:
        dict2file(voc, args.output)
    else:
        pprint(voc)

    debug_logger.debug('[Parse pairings] Ended')


if __name__ == '__main__':
    main()
