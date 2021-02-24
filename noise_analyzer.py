import argparse
import csv
import sys
from pathlib import Path

import terms


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('datafiles', action='store', type=str, nargs='+',
                        help='input TSV data files')
    parser.add_argument('--outfile', '-o', action='store', type=str,
                        default='-', help='output file in TSV')
    return parser


def main():
    args = init_argparser().parse_args()
    noise = {}
    for datafile in args.datafiles:
        tlist = terms.TermList()
        tlist.from_tsv(datafile)
        n_terms = tlist.get_from_label((terms.Label.NOISE,
                                        terms.Label.AUTONOISE,
                                        terms.Label.NONE))
        dataset = Path(datafile).stem
        for t in n_terms.items:
            try:
                noise[t.string]['count'] += 1
                noise[t.string]['datasets'].append(dataset)
                noise[t.string]['labels'].append(t.label.label_name)
            except KeyError:
                noise[t.string] = {'term': t.string, 'count': 1,
                                   'datasets': [dataset],
                                   'labels': [t.label.label_name]}

    # get the fieldnames directly from the objects
    fieldnames = next(iter(noise.values())).keys()

    if args.outfile == '-':
        out = sys.stdout
    else:
        out = open(args.outfile, 'w')

    with out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for v in noise.values():
            writer.writerow(v)


if __name__ == '__main__':
    main()
