import logging
import argparse
import os
import warnings

import pandas as pd
import numpy as np

from DocSim import DocSim
from gensim.models import Word2Vec, phrases, Phrases
from joblib import Parallel, delayed
import multiprocessing

from preprocess import load_stop_words
from utils import (
    assert_column,
    setup_logger,
    load_df,
)

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Creates words embeddings and computes a related dissimilarity metric')
    parser.add_argument('input', action="store", type=str,
                        help="input CSV data file with corpus processed")
    parser.add_argument('tdm', action="store", type=str,
                        help="input CSV data file with TDM")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name in CSV format')
    return parser


def load_vocabulary():
    voc = pd.read_csv('term-list.csv', sep='\t')
    return voc


def similarity(ds, source, targets, i):
    sim_scores = ds.calculate_similarity(source, targets)
    scores = [score['score'] for score in sim_scores]
    print(i)
    return scores


def main():
    debug_logger.debug('[Embeddings] Started')
    parser = init_argparser()
    args = parser.parse_args()

    #warnings.simplefilter(action='ignore', category=RuntimeWarning)

    voc = load_vocabulary()
    corpus = load_df(args.input, required_columns=['id', 'abstract_lem'])
    tabs = load_df('..\\RTS\\rts_terms_tab.csv', required_columns=['keyword'])
    stop_words = load_stop_words('..\\RTS\\stop_words.txt', language='english')
    tdm = pd.read_csv(args.tdm, sep='\t')
    dtm = tdm.T

    # use only documents that have not been skipped during post-processing
    # skip first index due to transposition
    corpus = corpus.iloc[dtm.index.values[1:]]

    train_sentences = []

    sentences_keywords = []
    sentences_keywords_counts = []
    for (idx, row) in dtm[1:].iterrows():
        nonzero = row[row != 0].index
        nonzero_counts = row[row != 0].values.tolist()
        voc_ = voc.iloc[nonzero, :]['term']
        sentences_keywords.append(voc_.values.tolist())
        sentences_keywords_counts.append(nonzero_counts)

    sent = [row.split() for row in corpus['abstract_lem']]
    bigrams = Phrases(sent, min_count=1, threshold=1, delimiter=b' ', common_terms=stop_words)
    trigrams = Phrases(bigrams[sent], min_count=1, threshold=1, delimiter=b' ', common_terms=stop_words)
    quadrigrams = Phrases(trigrams[bigrams[sent]], min_count=1, threshold=0.001, delimiter=b' ', common_terms=stop_words)

    for idx, s in enumerate(sent, 0):

        bigrams_ = [b for b in bigrams[s] if b.count(' ') == 1]
        trigrams_ = [t for t in trigrams[bigrams[s]] if t.count(' ') == 2]
        quadrigrams_ = [q for q in quadrigrams[trigrams[bigrams[s]]] if q.count(' ') == 3]

        for k in sentences_keywords[idx]:
            if k.count(' ') == 1:
                # bigram
                if not k in bigrams_:
                    bigrams_.append(k)
            elif k.count(' ') == 2:
                #trigram
                if not k in trigrams_:
                    trigrams_.append(k)
            elif k.count(' ') == 3:
                #quadigram
                if not k in quadrigrams_:
                    quadrigrams_.append(k)

        train_sentences.append(s + bigrams_ + trigrams_ + quadrigrams_)

    if not os.path.isfile('word2vec.model'):
        cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

        model = Word2Vec(train_sentences, size=100, window=10, min_count=0, iter=20,
                         alpha=0.03, min_alpha=0.0007, workers=cores-1)

        model.save('word2vec.model')

    else:

        model = Word2Vec.load('word2vec.model')
        ds = DocSim(model)

        num_cores = multiprocessing.cpu_count()
        print("Availables CPUs:", num_cores)

        s = None

        targets = sentences_keywords[:s]

        diss_matrix = Parallel(n_jobs=num_cores)(delayed(similarity)(ds, source, targets, i) for i, source in enumerate(targets, 0))

        df = pd.DataFrame(data=diss_matrix, index=corpus['id'].values.tolist()[:s], columns=corpus['id'].values.tolist()[:s])
        if args.output:
            df.to_csv(args.output, sep='\t', compression='gzip', index=True, header=True, float_format='%.5f', encoding='utf-8')
        else:
            print(df)


    debug_logger.debug('[Embeddings] Terminated')


if __name__ == '__main__':
    main()
