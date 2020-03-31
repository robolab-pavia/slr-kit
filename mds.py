import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from utils import (
    load_df,
    setup_logger,
)


debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Perform MDS over a distance matrix')

    parser.add_argument('infile', action="store", type=str,
                        help="input CSV file with documents-terms matrix")
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV file documents data (id, title, abstract)")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def do_PCA(dtm, n):
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the PCA algorithm with our Data
    pca = PCA(n_components=n).fit(data_rescaled)
    data_reducted = pca.transform(data_rescaled)

    dtm_reducted = pd.DataFrame(data_reducted, index=dtm.index)
    return dtm_reducted


def do_MDS(dtm, n):
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the MDS algorithm with our Data
    mds = MDS(n_components=n).fit(data_rescaled)
    data_reducted = mds.transform(data_rescaled)

    dtm_reducted = pd.DataFrame(data_reducted, index=dtm.index)
    return dtm_reducted


def compute_csimilarity(dtm):
    cs = cosine_similarity(dtm)
    cs_pd = pd.DataFrame(cs, index=dtm.index)
    return cs_pd


def do_clustering(dist_matrix, num_clusters):
    ward_cl = AgglomerativeClustering(n_clusters=num_clusters,
                                      affinity='euclidean',
                                      linkage='ward')
    return ward_cl.fit(dist_matrix)


def autotune_PCA(dtm):
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the PCA algorithm with our Data
    pca = PCA().fit(data_rescaled)

    sum = 0
    index = 0
    for var in pca.explained_variance_ratio_:
        if sum >= 0.90:
            break
        sum = sum + var
        index = index + 1

    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axvline(x=index, color='r', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='--')
    plt.plot(index, 0.90, marker='o', markersize=3, color="red")
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()


def select_n_components(var_ratio, goal_var: float) -> int:
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

        df = pd.DataFrame(dict(title=titles, label=ward_cl.labels_), index=dtm.index)

        output_file = "pca_" + str(n) + "_clusters_" + str(num_clusters) + ".csv"
        df.to_csv(output_file, header=True, sep='\t', encoding='utf-8')

        progress_bar.update(1)

    progress_bar.close()
    debug_logger.debug('[multidimensional scaling] Terminated')


if __name__ == "__main__":
    main()
