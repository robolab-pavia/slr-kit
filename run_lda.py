r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import argparse
import json
import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import Phrases, LdaModel
from gensim.models.coherencemodel import CoherenceModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    epilog = """"'The program uses two files: <dataset>/<prefix>_preproc.csv and
 '<dataset>/<prefix>_terms.csv.
 It outputs the topics in <dataset>/<prefix>_topics.json and the topics assigned
 to each document in <dataset>/<prefix>_docs-topics.json"""
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog=epilog)
    parser.add_argument('dataset', action="store", type=Path,
                        help='path to the directory where the files of the '
                             'dataset to elaborate are stored.')
    parser.add_argument('prefix', action="store", type=str,
                        help='prefix used when searching files.')
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

    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    titles = dataset['title'].to_list()
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in good_set]
        good_docs.append(gd)
    return good_docs, titles


def main():
    args = init_argparser().parse_args()

    terms_file = args.dataset / f'{args.prefix}_terms.csv'
    preproc_file = args.dataset / f'{args.prefix}_preproc.csv'
    docs, titles = generate_filtered_docs(terms_file, preproc_file)
    # good_docs = generate_raw_docs()
    # good_docs = good_docs[:2000]
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

    cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass',
                        processes=6)
    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    avg_topic_coherence = cm.get_coherence()
    print(f'Average topic coherence: {avg_topic_coherence:.4f}.')
    topics = {}
    coherence = cm.get_coherence_per_topic()
    for i in range(model.num_topics):
        topic = model.show_topic(i)
        t_dict = {
            'name': '',
            'terms_probability': {t[0]: float(t[1]) for t in topic},
            'coherence': float(coherence[i]),
        }
        topics[i] = t_dict

    topic_file = args.dataset / f'{args.prefix}_topics.json'
    with open(topic_file, 'w') as file:
        json.dump(topics, file, indent='\t')

    docs_topics = []
    for i, (title, d) in enumerate(zip(titles, docs)):
        bow = dictionary.doc2bow(d)
        t = model.get_document_topics(bow)
        d_t = {
            'id': i,
            'title': title,
            'topics': {tu[0]: float(tu[1]) for tu in t},
        }
        docs_topics.append(d_t)
        print(title, t)

    docs_file = args.dataset / f'{args.prefix}_docs-topics.json'
    with open(docs_file, 'w') as file:
        json.dump(docs_topics, file, indent='\t')


if __name__ == '__main__':
    main()
