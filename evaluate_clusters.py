import argparse
import itertools
import json
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from utils import (
    setup_logger,
)

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Compare cluster results with manually labeled documents')
    parser.add_argument('clusters_file', action="store", type=str,
                        help="input TSV file in format (id   label)")
    parser.add_argument('ground_truth', action="store", type=str,
                        help="input JSON file in format (id: [label1,label2,..]")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def labels_encoder(gt):
    voc = {}
    index = 0

    unique_labels = sorted(set([val[0] for val in gt.values()]))
    for label in unique_labels:
        voc.update({label: index})
        index += 1

    return voc


class PairingsEvaluator:

    class Stat(NamedTuple):
        label: str
        elements: int
        intersect_elements: int
        ratio: float

    def __init__(self, clusters, ground_truth):
        self.clusters = clusters
        self.ground_truth = ground_truth

        self.clusters_k = 0
        self.stats_data = []

    def evaluate(self):
        indexes = list(map(int, self.ground_truth['id'].values.tolist()))
        # reduce clusters_file matching only manually analyzed documents:
        reduced_df = self.clusters.loc[self.clusters['id'].isin(indexes), :]

        clusters_groups = reduced_df.groupby('label')
        self.clusters_k = len(clusters_groups)

        for label, df_group in clusters_groups:

            docs_in_cluster = df_group['id'].values.tolist()
            pairs_docs_in_cluster = all_pairs(docs_in_cluster)

            intersection_list = []

            for col in self.ground_truth.columns[1:]:

                # get pairs for this columns/label
                constraints = list(
                    map(int, self.ground_truth.loc[self.ground_truth[col] == 1, 'id'].values.tolist())
                )
                pairs_constraints = all_pairs(constraints)

                # find sets intersection (of pairs) between current cluster and ground_truth
                intersection = list(set(pairs_constraints) & set(pairs_docs_in_cluster))

                if len(intersection) > 0:
                    # concatenate with other labels from ground_truth
                    intersection_list += intersection

            # get rid of duplicates
            # intersection_list = list(dict.fromkeys(intersection_list))
            # with np.maximum prevent division by 0 for empty cluster
            ratio = len(intersection_list) / np.maximum(len(pairs_docs_in_cluster), 1e-10) * 100
            self.stats_data.append(self.Stat(label, df_group.shape[0], len(intersection_list), ratio))

    def stats(self):
        print("Automatic clustering: {} elements and k={} clusters".format(self.clusters.shape[0],
                                                                           self.clusters_k))

        print("Ground Truth: {} elements and m={} labels".format(self.ground_truth.shape[0],
                                                                 self.clusters.shape[1] - 1))  # exclude ID
        for stat in self.stats_data:
            print("\tCluster: ", stat.label, "[{}] elements".format(stat.elements),
                  f'matched {len(stat.intersect_elements)} unique pairs', "--> {:.2f} %".format(stat.ratio))


def parse_pairings(gt):
    """
    Transform the pairings into a DF with a sort of one-hot-encoding for each label assigned by the user
    and the document ID
    :param gt: path to pairings JSON file
    :return: pandas dataframe
    """

    fp = open(gt, 'r')
    ground_truth = json.load(fp)

    columns = sorted(set([val[0] for val in ground_truth.values()]))
    columns.insert(0, 'id')
    data = np.zeros([len(ground_truth.keys()), len(columns)], dtype=int)
    df = pd.DataFrame(data, columns=columns)
    df['id'] = ground_truth.keys()

    for doc in ground_truth:
        labels_list = ground_truth[doc]
        for label in labels_list:
            df.loc[df['id'] == doc, label] = 1
    return df


def all_pairs(partition):
    return list(itertools.combinations(partition, 2))


def main():

    parser = init_argparser()
    args = parser.parse_args()

    clusters_file = pd.read_csv(args.clusters_file, sep='\t')
    clusters_file.set_index(clusters_file['id'], inplace=True)

    ground_truth = parse_pairings(args.ground_truth)

    indexes = list(map(int, ground_truth['id'].values.tolist()))
    # reduce clusters_file matching only manually analyzed documents:
    reduced_df = clusters_file.loc[clusters_file['id'].isin(indexes), :]

    # Some documents may be missing?
    # main_list = np.setdiff1d(indexes, clusters_file['id'].values.tolist())
    # print(main_list)

    clusters_groups = reduced_df.groupby('label')

    print("Automatic clustering: {} elements and k={} clusters".format(clusters_file.shape[0], len(clusters_groups)))
    print("Ground Truth: {} elements and m={} labels".format(ground_truth.shape[0], ground_truth.shape[1] - 1))  # exclude ID
    for label, df_group in clusters_groups:

        docs_in_cluster = df_group['id'].values.tolist()
        pairs_docs_in_cluster = all_pairs(docs_in_cluster)

        intersection_list = []

        for col in ground_truth.columns[1:]:

            # get pairs for this columns/label
            constraints = list(map(int, ground_truth.loc[ground_truth[col] == 1, 'id'].values.tolist()))
            pairs_constraints = all_pairs(constraints)

            # find sets intersection (of pairs) between current cluster and ground_truth
            intersection = list(set(pairs_constraints) & set(pairs_docs_in_cluster))

            if len(intersection) > 0:

                # concatenate with other labels from ground_truth
                intersection_list += intersection

        # get rid of duplicates
        # intersection_list = list(dict.fromkeys(intersection_list))
        # with np.maximum prevent division by 0 for empty cluster
        ratio = len(intersection_list) / np.maximum(len(pairs_docs_in_cluster), 1e-10) * 100
        print("\tCluster: ", label, "[{}] elements".format(df_group.shape[0]),
              f'matched {len(intersection_list)} unique pairs', "--> {:.2f} %".format(ratio))


if __name__ == '__main__':
    main()