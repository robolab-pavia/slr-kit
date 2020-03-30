import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
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
    parser.add_argument('--samples', '-s', metavar='N',
                        help='number of samples into the dataset to use (default entire dataset)')
    parser.add_argument('--plot', '-p', metavar='DENDROGRAM_FILE',
                        help='save dendrogram graph of agglomerative clustering to a file')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def get_distances(X, model, mode='l2'):
    """
    AgglomerativeClustering from sklean does not provide distances and weights
    so this function computes them from a given model and input data
    :param X: distance matrix
    :param model: model from AgglomerativeClustering
    :param mode: parameter that deals with a higher level cluster merge with lower distance
    :return: distance and weight for each observation
    """
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d

        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


def plot_dendrogram(linkage_matrix, out_file, **kwargs):
    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(figsize=(15, 20))
    ax = dendrogram(linkage_matrix, **kwargs)

    # print vertical line to cut the graph
    # plt.axvline(x=22.5, color='r', linestyle='--')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    # uncomment below to save graph to file
    plt.savefig(out_file, dpi=300)
    plt.close()


def do_clustering(dist_matrix, num_clusters):
    ward_cl = AgglomerativeClustering(n_clusters=num_clusters,
                                      affinity='euclidean',
                                      linkage='ward')
    return ward_cl.fit(dist_matrix)


def main():
    debug_logger.debug('[hierarchical clustering] Started')
    parser = init_argparser()
    args = parser.parse_args()

    # dataset loading
    dist_matrix = pd.read_csv(args.infile, delimiter='\t', index_col=0)
    dist_matrix.fillna('', inplace=True)

    # default uses entire dataset
    max_docs = dist_matrix.shape[0]
    if args.samples is not None:
        max_docs = int(args.samples)

    dist_matrix = dist_matrix.iloc[0:max_docs, 0:max_docs]

    docs = load_df(args.datafile, required_columns=['id', 'title'])

    titles = docs.iloc[dist_matrix.index[0:max_docs].tolist()]['title']

    debug_logger.debug('[hierarchical clustering] Computing clusters')

    # selection of cluster by default or by argument
    num_clusters = default_n_clusters
    if args.clusters is not None:
        num_clusters = int(args.clusters)

    # performs Hierarchical Clustering using Ward's method
    ward_cl = do_clustering(dist_matrix, num_clusters)

    # retrieves cluster labels for each sample
    labels = ward_cl.labels_

    # samples distance and weight computation for linkage matrix
    distance, weight = get_distances(dist_matrix.values, ward_cl)

    # computes linkage matrix
    linkage_matrix = np.column_stack([ward_cl.children_,
                                      distance,
                                      weight]).astype(float)

    # Only for development: concatenates document Title and assigned Label
    labelled_titles = []
    for label, title in zip(labels, titles):
        labelled_titles.append("%s - %s" % (label, title))

    # if plotting is turned on
    if args.plot is not None:
        plot_dendrogram(linkage_matrix,
                        args.plot,
                        orientation="left",
                        labels=labelled_titles)

    df = pd.DataFrame(dict(title=titles, label=labels), index=dist_matrix.index)

    output_file = open(args.output, 'w', encoding='utf-8', newline='') if args.output is not None else sys.stdout
    export_csv = df.to_csv(output_file, header=True, sep='\t', encoding='utf-8')
    output_file.close()

    debug_logger.debug('[hierarchical clustering] Terminated')


if __name__ == "__main__":
    main()
