import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def do_PCA(tdm, n):

    dtm = tdm.T
    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    # Fitting the PCA algorithm with our Data
    pca = PCA(n_components=n).fit(data_rescaled)
    data_reducted = pca.transform(data_rescaled)

    dtm_reducted = pd.DataFrame(data_reducted)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    # plt.show()
    tdm_reducted = dtm_reducted.T
    return tdm_reducted


def cosine_similarity(tdm):
    dtm = tdm.T
    cs = cosine_similarity(dtm)
    cs_pd = pd.DataFrame(cs)
    return cs_pd

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

    # we want to use the documents-terms matrix so DONE IN FUNCTIONS
    # dtm = tdm.T

    n_components = tdm.shape[0]
    start = int(n_components*0.1)
    for n in range(start, n_components, int(n_components*0.1)):
        reduced_tdm = do_PCA(tdm, n)
        print(reduced_tdm.head())
        cs_tdm = cosine_similarity(reduced_tdm)
        print(cs_tdm.head())
        print(cs_tdm.shape)
    # performs MDS analysis
    # mds_ = doMDS(dtm)
    debug_logger.debug('[multidimensional scaling] Terminated')


if __name__ == "__main__":
    main()
