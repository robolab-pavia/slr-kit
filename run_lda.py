r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import argparse
import json
import logging
from itertools import repeat
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from utils import substring_index


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    epilog = """"'The program uses two files: <dataset>/<prefix>_preproc.csv and
 '<dataset>/<prefix>_terms.csv.
 It outputs the topics in <dataset>/<prefix>_terms-topics.json and the topics assigned
 to each document in <dataset>/<prefix>_docs-topics.json"""
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog=epilog)
    parser.add_argument('dataset', action='store', type=Path,
                        help='path to the directory where the files of the '
                             'dataset to elaborate are stored.')
    parser.add_argument('prefix', action='store', type=str,
                        help='prefix used when searching files.')
    parser.add_argument('--topics', action='store', type=int, default=20,
                        help='Number of topics. If omitted %(default)s is used')
    parser.add_argument('--alpha', action='store', type=str, default='auto',
                        help='alpha parameter of LDA. If omitted %(default)s is'
                             ' used')
    parser.add_argument('--beta', action='store', type=str, default='auto',
                        help='beta parameter of LDA. If omitted %(default)s is '
                             'used')
    parser.add_argument('--no_below', action='store', type=int, default=20,
                        help='Keep tokens which are contained in at least'
                             'this number of documents. If omitted %(default)s '
                             'is used')
    parser.add_argument('--no_above', action='store', type=float, default=0.5,
                        help='Keep tokens which are contained in no more than '
                             'this fraction of documents (fraction of total '
                             'corpus size, not an absolute number). If omitted '
                             '%(default)s is used')
    parser.add_argument('--ngrams', action='store_true',
                        help='if set use all the ngrams')
    parser.add_argument('--model', action='store_true',
                        help='if set the lda model is saved to directory '
                             '<dataset>/<prefix>_lda_model')
    parser.add_argument('--no-relevant', action='store_true',
                        help='if set, use only the term labelled as keyword')
    return parser


def load_ngrams(terms_file, labels=('keyword', 'relevant')):
    words_dataset = pd.read_csv(terms_file, delimiter='\t',
                                encoding='utf-8')
    terms = words_dataset['keyword'].to_list()
    term_labels = words_dataset['label'].to_list()
    zipped = zip(terms, term_labels)
    good = [x[0] for x in zipped if x[1] in labels]
    ngrams = {1: []}
    for x in good:
        n = x.count(' ') + 1
        try:
            ngrams[n].append(x)
        except KeyError:
            ngrams[n] = [x]

    return ngrams


def check_ngram(doc, idx):
    for dd in doc:
        r = range(dd[0][0], dd[0][1])
        yield idx[0] in r or idx[1] in r


def filter_doc(d, ngram_len, terms):
    doc = []
    flag = False
    for n in ngram_len:
        for t in terms[n]:
            for idx in substring_index(d, t):
                if flag and any(check_ngram(doc, idx)):
                    continue

                doc.append((idx, t.replace(' ', '_')))

        flag = True
    doc.sort(key=lambda dd: dd[0])
    return [t[1] for t in doc]


def load_terms(terms_file, labels=('keyword', 'relevant')):
    words_dataset = pd.read_csv(terms_file, delimiter='\t',
                                encoding='utf-8')
    terms = words_dataset['keyword'].to_list()
    term_labels = words_dataset['label'].to_list()
    zipped = zip(terms, term_labels)
    good = [x for x in zipped if x[1] in labels]
    good_set = set()
    for x in good:
        good_set.add(x[0])

    return good_set


def generate_filtered_docs_ngrams(terms_file, preproc_file,
                                  labels=('keyword', 'relevant')):
    terms = load_ngrams(terms_file, labels)
    ngram_len = sorted(terms, reverse=True)
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    titles = dataset['title'].to_list()
    docs = []
    with Pool() as pool:
        docs = pool.starmap(filter_doc, zip(documents, repeat(ngram_len),
                                            repeat(terms)))

    return docs, titles


def generate_filtered_docs(terms_file, preproc_file,
                           labels=('keyword', 'relevant')):
    terms = load_terms(terms_file, labels)
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    titles = dataset['title'].to_list()
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in terms]
        good_docs.append(gd)
    return good_docs, titles


def main():
    args = init_argparser().parse_args()

    terms_file = args.dataset / f'{args.prefix}_terms.csv'
    preproc_file = args.dataset / f'{args.prefix}_preproc.csv'

    if args.no_relevant:
        labels = ('keyword', )
    else:
        labels = ('keyword', 'relevant')

    if args.ngrams:
        docs, titles = generate_filtered_docs_ngrams(terms_file, preproc_file,
                                                     labels)
    else:
        docs, titles = generate_filtered_docs(terms_file, preproc_file, labels)

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
        chunksize = 10000
        passes = 20
        iterations = 600
        eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    _ = dictionary[0]  # This is only to "load" the dictionary.
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

    cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary,
                        coherence='c_v', processes=cpu_count())
    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    avg_topic_coherence = cm.get_coherence()
    coherence = cm.get_coherence_per_topic()
    print(f'Average topic coherence: {avg_topic_coherence:.4f}.')
    topics = {}
    topics_order = list(range(model.num_topics))
    topics_order.sort(key=lambda x: coherence[x], reverse=True)
    for i in topics_order:
        topic = model.show_topic(i)
        t_dict = {
            'name': f'Topic {i}',
            'terms_probability': {t[0]: float(t[1]) for t in topic},
            'coherence': f'{float(coherence[i]):.5f}',
        }
        topics[i] = t_dict

    topic_file = args.dataset / f'{args.prefix}_terms-topics.json'
    with open(topic_file, 'w') as file:
        json.dump(topics, file, indent='\t')

    docs_topics = []
    for i, (title, d) in enumerate(zip(titles, docs)):
        bow = dictionary.doc2bow(d)
        t = model.get_document_topics(bow)
        t.sort(key=lambda x: x[1], reverse=True)
        d_t = {
            'id': i,
            'title': title,
            'topics': {tu[0]: float(tu[1]) for tu in t},
        }
        docs_topics.append(d_t)

    docs_file = args.dataset / f'{args.prefix}_docs-topics.json'
    with open(docs_file, 'w') as file:
        json.dump(docs_topics, file, indent='\t')

    if args.model:
        lda_path: Path = args.dataset / f'{args.prefix}_lda_model'
        lda_path.mkdir(exist_ok=True)
        model.save(str(lda_path / 'model'))


if __name__ == '__main__':
    main()
