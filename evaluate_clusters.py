import argparse
import itertools
import json
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


class PairingsEvaluator:
    class Stat(NamedTuple):
        label: str
        elements: int
        intersect_elements: int
        ratio: float

    def __init__(self, clusters, ground_truth):
        self.clusters = clusters
        self.ground_truth = ground_truth

        self.__clusters_k = 0
        self.__stats_data = []

    def evaluate(self):
        indexes = list(map(int, self.ground_truth['id'].values.tolist()))
        # reduce clusters_file matching only manually analyzed documents:
        reduced_df = self.clusters.loc[self.clusters['id'].isin(indexes), :]

        clusters_groups = reduced_df.groupby('label')
        self.__clusters_k = len(clusters_groups)

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
            try:
                ratio = len(intersection_list) / len(pairs_docs_in_cluster) * 100
            except ZeroDivisionError:
                ratio = 0
            self.__stats_data.append(self.Stat(label, df_group.shape[0], len(intersection_list), ratio))

    def stats(self):
        print("Automatic clustering: {} elements and k={} clusters".format(self.clusters.shape[0],
                                                                           self.__clusters_k))

        print("Ground Truth: {} elements and m={} labels".format(self.ground_truth.shape[0],
                                                                 self.ground_truth.shape[1] - 1))  # exclude ID
        for stat in self.__stats_data:
            print("\tCluster: ", stat.label, "[{}] elements".format(stat.elements),
                  f'matched {stat.intersect_elements} unique pairs', "--> {:.2f} %".format(stat.ratio))


class ConfusionMatrixEvaluator:

    class Metric:

        def __init__(self, predicted_label, attended_label, TP, FP, TN, FN):
            self.predicted_label = predicted_label
            self.attended_label = attended_label
            self.TP = TP
            self.FP = FP
            self.TN = TN
            self.FN = FN

        def countTP(self):
            return len(self.TP)

        def countFP(self):
            return len(self.FP)

        def countTN(self):
            return len(self.TN)

        def countFN(self):
            return len(self.FN)

        def total(self):
            return self.countTN() + self.countFN() + self.countTP() + self.countFP()

        def precision(self):
            try:
                prec = self.countTP() / (self.countTP() + self.countFP())
            except ZeroDivisionError:
                prec = 0
            return "{:.2f}".format(prec)

        def accuracy(self):
            # prevent division by 0
            try:
                acc = (self.countTP() + self.countTN()) / self.total()
            except ZeroDivisionError:
                acc = 0
            return "{:.2f}".format(acc)

        def recall(self):
            try:
                recall = self.countTP() / (self.countTP() + self.countFN())
            except ZeroDivisionError:
                recall = 0
            return "{:.2f}".format(recall)

        def f1_score(self):
            try:
                f1 = (2*self.countTP()) / (2*self.countTP() + self.countFP() + self.countFN())
            except ZeroDivisionError:
                f1 = 0
            return "{:.2f}".format(f1)

        def __str__(self):
            return format_matrix([f'in {self.predicted_label}', f'not in {self.predicted_label}'],
                                 [[self.countTP(), self.countFP()], [self.countFN(), self.countTN()]],
                                 '{:^{}}', '{:<{}}', '{:>{}.3f}', '\n', ' | ')

    def __init__(self, clusters, ground_truth):
        self.clusters = clusters
        self.ground_truth = ground_truth

        self.__CM = []

    def evaluate(self):
        indexes = list(map(int, self.ground_truth['id'].values.tolist()))
        # reduce clusters_file matching only manually analyzed documents:  -------->   TOT
        reduced_df = self.clusters.loc[self.clusters['id'].isin(indexes), :]

        TOT = set(reduced_df['id'].values.tolist())

        clusters_groups = reduced_df.groupby('label')

        for label, df_group in clusters_groups:
            docs_in_cluster = df_group['id'].values.tolist()

            row = []
            for col in self.ground_truth.columns[1:]:
                constraints = list(
                    map(int, self.ground_truth.loc[self.ground_truth[col] == 1, 'id'].values.tolist())
                )

                A = set(docs_in_cluster)
                B = set(constraints)

                TP = list(A & B)
                FP = list(A - (A & B))
                TN = list((TOT - A) & (TOT - B))
                FN = list(B - A)

                row.append(self.Metric(predicted_label=str(label), attended_label=col, TP=TP, FP=FP, TN=TN, FN=FN))

            self.__CM.append(row)

    def stats(self):

        df_acc = pd.DataFrame(columns=self.ground_truth.columns[1:])
        df_rec = pd.DataFrame(columns=self.ground_truth.columns[1:])
        df_f1 = pd.DataFrame(columns=self.ground_truth.columns[1:])
        df_prec = pd.DataFrame(columns=self.ground_truth.columns[1:])

        for row in self.__CM:
            acc_row = {}
            rec_row = {}
            f1_row = {}
            prec_row = {}
            for m in row:
                acc_row.update({m.attended_label: float(m.accuracy())})
                rec_row.update({m.attended_label: float(m.recall())})
                f1_row.update({m.attended_label: float(m.f1_score())})
                prec_row.update({m.attended_label: float(m.precision())})

            df_acc = df_acc.append(acc_row, ignore_index=True)
            df_rec = df_rec.append(rec_row, ignore_index=True)
            df_f1 = df_f1.append(f1_row, ignore_index=True)
            df_prec = df_prec.append(prec_row, ignore_index=True)

        plt.rcParams["figure.figsize"] = (15, 10)

        df_acc.to_csv(f'eval_accuracy.csv', sep='\t')

        plt.figure()
        chart = sns.heatmap(df_acc, cmap='coolwarm', linewidths=0.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right', fontweight='light')
        plt.title("Accuracy")
        plt.ylabel("Automatic clusters")
        plt.xlabel("Manually assigned clusters")
        plt.savefig('accuracy.png', dpi=100)

        df_rec.to_csv(f'eval_recall.csv', sep='\t')
        plt.figure()
        chart = sns.heatmap(df_rec, cmap='coolwarm', linewidths=0.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right', fontweight='light')
        plt.title("Recall")
        plt.ylabel("Automatic clusters")
        plt.xlabel("Manually assigned clusters")
        plt.savefig('recall.png', dpi=100)

        df_rec.to_csv(f'eval_f1.csv', sep='\t')
        plt.figure()
        chart = sns.heatmap(df_rec, cmap='coolwarm', linewidths=0.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right', fontweight='light')
        plt.title("F1-score")
        plt.ylabel("Automatic clusters")
        plt.xlabel("Manually assigned clusters")
        plt.savefig('f1_score.png', dpi=100)

        df_prec.to_csv(f'eval_precision.csv', sep='\t')
        plt.figure()
        chart = sns.heatmap(df_prec, cmap='coolwarm', linewidths=0.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right', fontweight='light')
        plt.title("Precision")
        plt.ylabel("Automatic clusters")
        plt.xlabel("Manually assigned clusters")
        plt.savefig('precision.png', dpi=100)



def labels_encoder(gt):
    voc = {}
    index = 0

    unique_labels = sorted(set([val[0] for val in gt.values()]))
    for label in unique_labels:
        voc.update({label: index})
        index += 1

    return voc


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


def format_matrix(header, matrix, top_format, left_format, cell_format, row_delim, col_delim):
    table = [[''] + header] + [[name] + row for name, row in zip(header, matrix)]
    table_format = [['{:^{}}'] + len(header) * [top_format]] \
                 + len(matrix) * [[left_format] + len(header) * [cell_format]]
    col_widths = [max(
                      len(format.format(cell, 0))
                      for format, cell in zip(col_format, col))
                  for col_format, col in zip(zip(*table_format), zip(*table))]
    return row_delim.join(
               col_delim.join(
                   format.format(cell, width)
                   for format, cell, width in zip(row_format, row, col_widths))
               for row_format, row in zip(table_format, table))


def main():
    parser = init_argparser()
    args = parser.parse_args()

    clusters_file = pd.read_csv(args.clusters_file, sep='\t')
    clusters_file.set_index(clusters_file['id'], inplace=True)

    ground_truth = parse_pairings(args.ground_truth)

    # Evaluate pairs constraints
    pairs = PairingsEvaluator(clusters=clusters_file, ground_truth=ground_truth)
    pairs.evaluate()
    pairs.stats()

    # Compute Confusion Matrix
    cm = ConfusionMatrixEvaluator(clusters=clusters_file, ground_truth=ground_truth)
    cm.evaluate()
    cm.stats()


if __name__ == '__main__':
    main()
