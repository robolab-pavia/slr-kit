import logging
import argparse
import json
import pandas as pd
import numpy as np
from utils import (
    setup_logger,
    load_df,
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


import itertools


def all_pairs(partition):

    return list(itertools.combinations(partition, 2))


def main():

    parser = init_argparser()
    args = parser.parse_args()

    clusters_file = pd.read_csv(args.clusters_file, sep='\t')
    clusters_file.set_index(clusters_file['id'], inplace=True)

    ground_truth = parse_pairings(args.ground_truth)

    clusters_groups = clusters_file.groupby('label')

    for label, df_group in clusters_groups:

        docs_in_cluster = df_group['id'].values.tolist()
        pairs_docs_in_cluster = all_pairs(docs_in_cluster)

        count = 0
        total = 0

        # f = open(f'C:\\Users\\Marco\\Desktop\\cluster_{label}.txt', 'w')
        #
        # for d in docs_in_cluster:
        #     f.write(str(d) + '\n')
        #
        # f.write('\nDOCUMENTI ETICHETTATI:\n')

        intersection_list = []

        for col in ground_truth.columns[1:]:

            constraints = list(map(int, ground_truth.loc[ground_truth[col] == 1, 'id'].values.tolist()))
            pairs_constraints = all_pairs(constraints)
            intersection = list(set(pairs_constraints) & set(pairs_docs_in_cluster))

            if len(intersection) > 0:

                for val in intersection:
                    intersection_list.append(val)
                # how many docs of the constraints are actually in cluster
                #
                # count += len(intersection)
                total += len(pairs_constraints)
                # for c in intersection:
                #     f.write(str(c) + '\n')

        intersection_list = list(dict.fromkeys(intersection_list))
        ratio = len(intersection_list) / total * 100
        print("Cluster: ", label, "[{}] elements".format(df_group.shape[0]), f'matched {len(intersection_list)} unique pairs over {total}', "--> {:.2f} %".format(ratio))

        # print("\tVincolo su", col, "rispettato al ", "{:.2f}%".format(ratio), " --> W:{:.2f}".format(W),
        #       " --> {:.2f}".format(W * ratio))

    print("Number of constraints:", ground_truth.shape[0])
        # f.close()

    # scan clusterization results, doc by doc
    # for label, df_group in clusters_groups:
    #     docs_in_cluster = df_group['id'].values.tolist()
    #     print(docs_in_cluster)
    #     # here we have a dataframe for each cluster
    #     count = 0
    #     total = 0
    #     for (columnName, columnData) in ground_truth.iteritems():
    #         if columnName != 'id':
    #             togethers = ground_truth.loc[ground_truth[columnName] == 1, 'id']
    #             total += len(togethers.values.tolist())
    #             count += len(list(set(map(int, togethers.values.tolist())) & set(docs_in_cluster)))
    #     overlapping = count/total * 100
    #     print("For the label:", label, " the constraints have been respected for the {:.2f}%".format(overlapping))


if __name__ == '__main__':
    main()