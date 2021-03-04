import argparse
import itertools
import json
import logging

import pandas as pd
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans
from sklearn import metrics

from utils import load_df, setup_logger

debug_logger = setup_logger('debug_logger', 'slr-kit.log', level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='SemiSupervised Clustering with pairwise constraints')
    parser.add_argument('input', action="store", type=str, metavar='GROUND_TRUTH',
                        help="input JSON with ground truth information: id:[label1, label2, ..]")
    parser.add_argument('matrix', action="store", type=str, metavar='SIM_MATRIX',
                        help="precomputed similarity matrix")
    parser.add_argument('datafile', action="store", type=str, metavar='DATA',
                        help="input CSV file with documents ID and titles")
    parser.add_argument('--output', '-o', metavar='FILENAME', help='output file name in CSV format with clusters')
    return parser


def ground_truth(pairings_file):
    """ Parse pairings file and create a DataFrame with the ground truth """
    labels = {}

    # create a dictionary assigning an ID to a label
    with open(pairings_file) as json_file:
        data = json.load(json_file)

        id = 0
        for doc in data:
            for label in data[doc]:
                try:
                    l = labels[label]
                except KeyError:
                    labels.update({label: id})
                    id += 1

    # sort by ID
    labels = {v: k for k, v in sorted(labels.items(), key=lambda item: item[1])}

    gt_labels = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])

    # create dataframe for the ground truth

    docs = []
    labels_ = []

    with open(pairings_file) as json_file:
        data = json.load(json_file)

        for doc in data:
            label = data[doc][0]  # select first
            # for label in data[doc]:
            label_id = gt_labels.loc[gt_labels['label'] == label].index.values.tolist()[0]

            docs.append(doc)
            labels_.append(label_id)

    df = pd.DataFrame(data={'id': docs, 'label': labels_}, index=docs)

    return df


def all_pairs(partition):
    return list(itertools.combinations(partition, 2))


def gt_constraints(gt, row2id, max=None):
    """
    Compute MustLink and CannotLink constraints to seed clustering algorithm
    :param gt: Pandas Dataframe with ground truth (id, label)
    :param row2id: dictionary to handle missing document with theirs ID
    :param max: if set, is the maximum number of document to use to build the constraints
    :return: [ml, cl]
    """

    ml = []
    cl = []

    for doc in gt['id']:
        try:
            gt.loc[gt['id'] == doc, 'id'] = str(row2id[int(doc)])
        except KeyError:
            pass

    gt_groups = gt.groupby('label')

    # MUST LINK
    for label, df_group in gt_groups:
        docs_in_cluster = df_group['id'].values.tolist()[:max]
        ml += all_pairs(map(int, docs_in_cluster))

    # CANNOT LINK
    for label, df_group in gt_groups:

        docs_in_cluster = list(map(int, df_group['id'].values.tolist()[:max]))

        for doc in docs_in_cluster:
            for label1, df_group1 in gt_groups:

                if label == label1:
                    continue

                docs_in_cluster1 = list(map(int, df_group1['id'].values.tolist()[:max]))

                # all documents of two different clusters
                pairs = []
                for doc1 in docs_in_cluster1:
                    pairs += all_pairs([doc] + [doc1])

                cl += pairs

    return ml, cl


def check_results(gt, df):
    # get assigned labels and documents ID
    y = list(map(int, gt['label'].values.tolist()))
    y_id = list(map(int, gt['id'].values.tolist()))

    labels = df[df['id'].isin(y_id)]['label'].values.tolist()

    if len(y) == len(labels):
        # if ARI == 1 then all constraints have been respected
        ARI = metrics.adjusted_rand_score(y, labels)

    else:
        # something went wrong
        ARI = -1

    return ARI


def main():
    debug_logger.debug('[semi-supervised clustering] Started')
    parser = init_argparser()
    args = parser.parse_args()

    gt = ground_truth(args.input)

    # loads similarity matrix
    X = pd.read_csv(args.matrix, sep='\t', index_col=0)

    docs = load_df(args.datafile, required_columns=['id', 'title'])
    ids = docs.iloc[X.index.tolist()]['id'].values
    titles = docs.iloc[X.index.tolist()]['title'].values

    # since some documents are missing, store relationship between row and ID
    id2row = {}

    for row, id in zip(range(X.shape[0]), map(int, X.index.values.tolist())):
        id2row.update({id: row})

    # clustering algorithms works using row number as entity index, thus we reset
    # the index and we will get the association row-id back later
    X.reset_index(inplace=True, drop=True)
    X = X.to_numpy()

    debug_logger.debug('[semi-supervised clustering] Creating constraints')
    # ml = MUST LINK, cl = CANNOT LINK
    gt_ = gt.copy()
    ml, cl = gt_constraints(gt_, id2row)
    pairwise_constraints = [ml, cl]

    # perform clustering
    cls = COPKMeans(n_clusters=gt['label'].nunique())
    cls.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])

    # save output
    debug_logger.debug('[semi-supervised clustering] Clustering')
    df = pd.DataFrame(data={'id': ids, 'title': titles, 'label': cls.labels_}, columns=['id', 'title', 'label'])
    df.set_index(ids, inplace=True)
    df.to_csv(args.output, sep='\t', index=False, encoding='utf-8')

    # check ARI to see how much constraints have been respected
    ARI = check_results(gt, df)
    print("Adjusted Rand Index [0-1]:", ARI)

    debug_logger.debug('[semi-supervised clustering] Finished')


if __name__ == '__main__':
    main()
