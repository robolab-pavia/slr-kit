import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from utils import (
    load_df,
    setup_logger,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Perform Multidimensional Scaling (MDS) over a distance matrix')

    parser.add_argument('infile', action="store", type=str,
                        help="input CSV file with documents-terms matrix")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser

def doPCA(dtm):
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
    tdm.fillna('', inplace=True)

    # we want to use the documents-terms matrix so
    dtm = tdm.T

    # performs PCA analysis
    # pca_ = doPCA(dtm)

    # scale dtm in range [0:1] to better variance maximization
    scl = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scl.fit_transform(dtm)

    tsvd = TruncatedSVD(n_components=data_rescaled.shape[1] - 1)
    X_tsvd = tsvd.fit(data_rescaled)

    # List of explained variances
    tsvd_var_ratios = tsvd.explained_variance_ratio_

    optimal_components = select_n_components(tsvd_var_ratios, 0.90)


    # TODO: allow the selection of the filename from command line
    terms = load_df('term-list.csv', required_columns=['id', 'term'])

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    debug_logger.debug('[multidimensional scaling] Calculating distances information')

    U, Sigma, VT = randomized_svd(data_rescaled, n_components=optimal_components, n_iter=100, random_state=122)

    # for i, comp in enumerate(VT):
    #     terms_comp = zip(terms['term'], comp)
    #     print(comp)
    #     sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:]
    #     print("Concept " + str(i) + ": ")
    #     for t in sorted_terms:
    #         print(t[0] , end=",")
    #     print(" ")

    U_df = pd.DataFrame(U)
    print(U_df.head())
    debug_logger.debug('[multidimensional scaling] Saving')
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    export_csv = U_df.to_csv(output_file, header=True, sep='\t')
    output_file.close()

    debug_logger.debug('[multidimensional scaling] Terminated')


if __name__ == "__main__":
    main()
