r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import argparse
import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import Phrases, LdaModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog='The program uses two files: '
                                            '<dataset>/<prefix>_preproc.csv and '
                                            '<dataset>/<prefix>_terms.csv')
    parser.add_argument('dataset', action="store", type=Path,
                        help='path to the directory where the files of the '
                             'dataset to elaborate are stored.')
    parser.add_argument('prefix', action="store", type=str,
                        help='prefix used when searching files.')

    parser.add_argument('--no-relevant', '-n', action='store_true',
                        help='Do not use the relevant term. Use only keywords')
    parser.add_argument('--outfile', '-o', action="store", default='-',
                        type=argparse.FileType('w'),
                        help="output CSV data file")
    return parser


def generate_raw_docs():
    dataset = pd.read_csv('rts_preproc.csv', delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    docs = [d.split(' ') for d in documents]
    return docs


def generate_filtered_docs(terms_file, preproc_file):
    words_dataset = pd.read_csv(terms_file, delimiter='\t',
                                encoding='utf-8')
    terms = words_dataset['keyword'].to_list()
    labels = words_dataset['label'].to_list()
    zipped = zip(terms, labels)
    good = [x for x in zipped if x[1] == 'keyword' or x[1] == 'relevant']
    good_set = set()
    for x in good:
        good_set.add(x[0])
    # print(good_set)
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in good_set]
        good_docs.append(gd)
    return good_docs


def main():
    args = init_argparser().parse_args()

    terms_file = args.dataset / f'{args.prefix}_terms.csv'
    preproc_file = args.dataset / f'{args.prefix}_preproc.csv'
    good_docs = generate_filtered_docs(terms_file, preproc_file)
    # good_docs = generate_raw_docs()
    # good_docs = good_docs[:2000]
    docs = good_docs
    # sys.exit(1)

    # Compute bigrams.
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # We remove rare words and common words based on their *document frequency*.
    # Below we remove words that appear in less than 20 documents or in more than
    # 50% of the documents. Consider trying to remove words only based on their
    # frequency, or maybe combining that with this approach.
    #

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of
    # the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Finally, we transform the documents to a vectorized form. We simply
    # compute the frequency of each word, including the bigrams.

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Let's see how many tokens and documents we have to train on.

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Train LDA model.
    # Set training parameters.
    if False:
        num_topics = 20
        chunksize = 4000
        passes = 20
        iterations = 600
        eval_every = None  # Don't evaluate model perplexity, takes too much time.
    else:
        num_topics = 30
        chunksize = 2000
        passes = 20
        iterations = 4
        eval_every = 5  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(corpus)  # , num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)

    for d in docs:
        bow = dictionary.doc2bow(d)
        t = model.get_document_topics(bow)
        print(t)


if __name__ == '__main__':
    main()
