import argparse
import logging
import sys

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from utils import (
    load_df,
    setup_logger
)

default_n_clusters = 5

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Perform a hierarchical clustering using Ward algorithm')

    parser.add_argument('infile', action="store", type=str,
                        help="input CSV file with a computed distances matrix")
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV file documents data (id, title, abstract)")
    parser.add_argument('--clusters', '-n', metavar='N',
                        help='number of clusters to use (default n={})'.format(default_n_clusters))
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def main():
    debug_logger.debug('[hierarchical clustering] Started')
    parser = init_argparser()
    args = parser.parse_args()

    # dataset loading
    dist_matrix = pd.read_csv(args.infile, delimiter='\t', index_col=0)
    dist_matrix.fillna('', inplace=True)

    docs = load_df(args.datafile, required_columns=['id', 'title'])

    debug_logger.debug('[hierarchical clustering] Computing clusters')

    num_clusters = default_n_clusters
    if args.clusters is not None:
        num_clusters = int(args.clusters)

    ward_cl = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    ward_cl.fit(dist_matrix)
    labels = ward_cl.labels_

    # linkage_matrix = ward(dist_matrix)  # define the linkage_matrix using ward clustering
    #
    # # obtain clusters from the linkage matrix setting the max number
    # # of clusters to obtain equal to 'num_clusters'
    # labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    df = pd.DataFrame(dict(title=docs['title'], label=labels), index=dist_matrix.index)

    output_file = open(args.output, 'w', encoding='utf-8') if args.output is not None else sys.stdout
    export_csv = df.to_csv(output_file, header=True, sep='\t', encoding='utf-8')
    output_file.close()

    debug_logger.debug('[hierarchical clustering] Terminated')


if __name__ == "__main__":
    main()
