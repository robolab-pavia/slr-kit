from sklearn.cluster import KMeans
import pandas as pd
import sys
import logging
import argparse
from utils import (
    assert_column,
    setup_logger,
    load_df,
)

default_n_clusters = 5

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Clusterize with k-means the rows in the cosine similarity matrix.')
    parser.add_argument('matrix', action="store", type=str,
                        help="input CSV data file with cosine similarity matrix")
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV file with documents data (id, title, abstract)")
    parser.add_argument('--clusters', '-n', metavar='N',
                        help='number of clusters to use (default n={})'.format(default_n_clusters))
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def report_count(clusters):
    count = [[x,clusters.count(x)] for x in set(clusters)]
    for c in count:
        print('Cluster {}: {} terms'.format(c[0], c[1]))


def main():
    debug_logger.debug('[kmeans] Started')
    parser = init_argparser()
    args = parser.parse_args()

    # load the dataset
    debug_logger.debug('[kmeans] Loading input file')
    df = pd.read_csv(args.matrix, delimiter='\t', index_col=0, encoding='utf-8')
    df.fillna('', inplace=True)

    # fix indexes for missing documents
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    df.index.name = None  # drop 'Unnamed: 0' coming from transposition

    docs = load_df(args.datafile, required_columns=['id', 'title'])

    num_clusters = default_n_clusters
    if args.clusters is not None:
        num_clusters = int(args.clusters)

    debug_logger.debug('[kmeans] Clustering')
    km = KMeans(n_clusters=num_clusters)
    km.fit(df)
    clusters = km.labels_.tolist()

    cs_pd = pd.DataFrame(dict(title=docs['title'], label=clusters), index=df.index)

    ## write to output, either a file or stdout (default)
    debug_logger.debug('[kmeans] Saving')
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = cs_pd.to_csv(output_file, header=True, sep='\t', float_format='%.3f')
    output_file.close()

    debug_logger.debug('[kmeans] Terminated')


if __name__ == "__main__":
    main()

















