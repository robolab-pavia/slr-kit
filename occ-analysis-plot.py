import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import load_df


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', action="store", type=str,
                        help="input CSV data file")
    # parser.add_argument('--no-profile', action='store_true', dest='no_profile',
    #                     help='disable profiling logging')
    # parser.add_argument('--output', '-o', metavar='FILENAME',
    #                     help='output file name in line-wise JSON format')
    return parser


def main():
    args = init_argparser().parse_args()
    cols = ['abstract', 'keyword_count', 'not_relevant_count', 'keyword',
            'not-relevant']
    df = load_df(args.infile, required_columns=cols)
    max_kw = df['keyword_count'].max() + 1
    max_nrel = df['not_relevant_count'].max() + 1
    counts = df.apply(lambda row: (row['keyword_count'],
                                   row['not_relevant_count']),
                      axis=1)
    counts: pd.Series = counts.value_counts()
    x = []
    y = []
    vals = []
    for idx, count in counts.items():
        x.append(idx[0])
        y.append(idx[1])
        vals.append(count)

    vals = np.array(vals)

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, s=vals, c=vals, cmap='inferno')
    cbar = ax.figure.colorbar(sc, ax=ax)
    plt.xticks(range(1, max(x)+3, 3))
    plt.yticks(range(1, max(y)+3, 3))
    plt.xlabel('N° of keyword')
    plt.ylabel('N° of not relevant')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
