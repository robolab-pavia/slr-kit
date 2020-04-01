import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from utils import (
    load_df,
    setup_logger,
)

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(
        description='Perform dimensionality reduction over a documents-terms matrix')

    parser.add_argument('infile', action="store", type=str,
                        help="input CSV file with documents-terms matrix")
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV file documents data (id, title, abstract)")
    parser.add_argument('--smart', '-S', action='store_true', dest='S',
                        help="Enable SMART search")
    parser.add_argument('--variance', '-v', action="store", metavar="VARIANCE", type=float,
                        dest='v', help="Variance to retain with SMART search (default 90%%)")
    parser.add_argument('--plot', '-p', metavar="FILENAME", type=str,
                        help="output file with global variance over components plot: only for SMART search")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def do_PCA(dtm, n):
    """
    Performs PCA reduction over a documents-terms matrix
    :param dtm: input documents-terms matrix
    :param n: desired number to dimensions
    :return: dtm with n features
    """

    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the PCA algorithm with our Data
    pca = PCA(n_components=n).fit(data_rescaled)
    data_reducted = pca.transform(data_rescaled)

    dtm_reducted = pd.DataFrame(data_reducted, index=dtm.index)
    return dtm_reducted


def do_MDS(dtm, n):
    """
    Performs MDS reduction over a documents-terms matrix
    :param dtm: input documents-terms matrix
    :param n: desired number to dimensions
    :return: dtm with n features
    """
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the MDS algorithm with our Data
    mds = MDS(n_components=n).fit(data_rescaled)
    data_reducted = mds.transform(data_rescaled)

    dtm_reducted = pd.DataFrame(data_reducted, index=dtm.index)
    return dtm_reducted


def compute_csimilarity(dtm):
    """
    Compute cosine similarity for a given documents-terms matrix
    :param dtm: input documents-terms matrix (docs x features)
    :return: return a (docs x docs) distance matrix
    """
    cs = cosine_similarity(dtm)
    cs_pd = pd.DataFrame(cs, index=dtm.index)
    return cs_pd


def do_clustering(dist_matrix, num_clusters):
    """
    Compute HAC with Ward's linkage
    :param dist_matrix: input distance matrix
    :param num_clusters: desired number of clusters
    :return: AgglomerativeClustering model
    """
    ward_cl = AgglomerativeClustering(n_clusters=num_clusters,
                                      affinity='euclidean',
                                      linkage='ward')
    return ward_cl.fit(dist_matrix)


def autotuning_PCA(dtm, var=0.90, plot=False):
    """
    Find the optimum number of dimensions such that the 90% of
    the global variance is maintained
    :param dtm: input documents-terms matrix
    :param var: global variance to retain (default: 0.90)
    :param plot: if setted, plot to this file variance over components
    :return: optimal_components, dtm_reducted
    """
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the PCA algorithm with our Data
    pca = PCA().fit(data_rescaled)
    # get optimum number of components
    optimal_components = select_n_components(pca.explained_variance_ratio_,
                                             var)

    # use optimum number of components
    dtm_reducted = do_PCA(dtm, optimal_components)

    if plot:
        # Plotting the Cumulative Summation of the Explained Variance
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.axvline(x=optimal_components, color='r', linestyle='--')
        plt.axhline(y=var, color='r', linestyle='--')
        plt.xticks([0, optimal_components, dtm.shape[1]])
        plt.yticks([0, 0.5, 1, var])
        plt.plot(optimal_components, var, marker='o',
                 markersize=5, color='r', mec='r', mfc='w')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title('Explained Variance')
        plt.savefig(plot, dpi=300)

    return optimal_components, dtm_reducted


def select_n_components(var_ratio, goal_var: float) -> int:
    """
    Returns the number of components with which the goal_varù
    is retained
    :param var_ratio: Array of variance ratio for pca components
    :param goal_var: desired global variance to retain
    :return: number of components
    """
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components


def append_id(out_filename, n_components, n_clusters):
    """
    Append to user's specified filename information about
    components and clusters
    """
    filename = os.path.basename(out_filename)
    file, ext = os.path.splitext(filename)
    return "{0}_pca_{1}_clusters_{2}.{3}".format(file,
                                                 n_components,
                                                 n_clusters,
                                                 ext)


def main():
    debug_logger.debug('[multidimensional scaling] Started')
    parser = init_argparser()
    args = parser.parse_args()

    tdm = pd.read_csv(args.infile, delimiter='\t', index_col=0)
    tdm.fillna('0', inplace=True)

    dtm = tdm.T
    # preserve index as docs_id

    docs = load_df(args.datafile, required_columns=['id', 'title'])
    titles = docs.iloc[dtm.index.tolist()]['title'].values

    num_clusters = 5

    if not args.S:
        # opt1: grid search in range of available dimensions
        debug_logger.debug('[multidimensional scaling] Started grid-search')

        n_components = tdm.shape[0]
        start = int(n_components*0.2)
        step = start

        progress_bar = tqdm.tqdm(total=len(range(start, n_components, step)))

        for n in range(start, n_components, step):
            progress_bar.set_description("Performing PCA")
            progress_bar.refresh()
            reduced_dtm = do_PCA(dtm, n)

            progress_bar.set_description("Computing cosine similarity")
            progress_bar.refresh()
            dist_matrix = compute_csimilarity(reduced_dtm)

            progress_bar.set_description("Clustering with Ward HAC")
            progress_bar.refresh()
            ward_cl = do_clustering(dist_matrix, num_clusters)

            progress_bar.set_description("Saving results")
            progress_bar.refresh()

            df = pd.DataFrame(dict(title=titles, label=ward_cl.labels_),
                              index=dtm.index)

            if args.output:
                output_file = append_id(args.output, n, num_clusters)
                df.to_csv(output_file, header=True, sep='\t', encoding='utf-8')
            else:
                out_str = "PCA: {0} components - HAC: {1} clusters".format(n,
                                                                           num_clusters)
                print(out_str)
                print(df)

            progress_bar.update(1)

        progress_bar.close()

    else:
        # opt 2: automatic seach best model with
        # 90% of variance retained
        debug_logger.debug('[multidimensional scaling] Started autotuning PCA')

        var = 0.90
        plot = False
        if args.v:
            var = args.v
        if args.plot:
            plot = args.plot

        n, reduced_dtm = autotuning_PCA(dtm, var, plot)   # PCA
        dist_matrix = compute_csimilarity(reduced_dtm)  # COSINE SIMILARITY
        ward_cl = do_clustering(dist_matrix, num_clusters)  # HAC

        df = pd.DataFrame(dict(title=titles, label=ward_cl.labels_), index=dtm.index)

        if args.output:
            output_file = append_id(args.output, n, num_clusters)
            df.to_csv(output_file, header=True, sep='\t', encoding='utf-8')
        else:
            out_str = "PCA: {0} components - HAC: {1} clusters".format(n , num_clusters)
            print(out_str)
            print(df)

    debug_logger.debug('[multidimensional scaling] Terminated')


if __name__ == "__main__":
    main()
