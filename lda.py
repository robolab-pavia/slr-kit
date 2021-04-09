r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import argparse
import json
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from lda_utils import PHYSICAL_CPUS, load_documents
from utils import (substring_index, AppendMultipleFilesAction,
                   BARRIER_PLACEHOLDER)


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
    parser.add_argument('--additional-terms', '-T',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='additional_file',
                        help='Additional keyword file name')
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
    parser.add_argument('--seed', type=int, help='Seed to be used in training')
    parser.add_argument('--ngrams', action='store_true',
                        help='if set use all the ngrams')
    parser.add_argument('--model', action='store_true',
                        help='if set the lda model is saved to directory '
                             '<dataset>/<prefix>_lda_model. The model is saved '
                             'with name "model.')
    parser.add_argument('--no-relevant', action='store_true',
                        help='if set, use only the term labelled as keyword')
    parser.add_argument('--load-model', action='store',
                        help='Path to a directory where a previously trained '
                             'model is saved. Inside this directory the model '
                             'named "model" is searched. the loaded model is '
                             'used with the dataset file to generate the topics'
                             ' and the topic document association')
    parser.add_argument('--placeholder', '-p', default=BARRIER_PLACEHOLDER,
                        help='Placeholder for barrier word. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)s')
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
                                  labels=('keyword', 'relevant'),
                                  additional=None,
                                  barrier_placeholder=BARRIER_PLACEHOLDER,
                                  relevant_prefix=BARRIER_PLACEHOLDER):
    if additional is None:
        additional = set()

    terms = load_ngrams(terms_file, labels) | additional
    ngram_len = sorted(terms, reverse=True)
    target_col = 'abstract_lem'
    documents, titles = load_documents(preproc_file, target_col,
                                       barrier_placeholder, relevant_prefix)
    with Pool(processes=PHYSICAL_CPUS) as pool:
        docs = pool.starmap(filter_doc, zip(documents, repeat(ngram_len),
                                            repeat(terms)))

    return docs, titles


def generate_filtered_docs(terms_file, preproc_file,
                           labels=('keyword', 'relevant'), additional=None,
                           barrier_placeholder=BARRIER_PLACEHOLDER,
                           relevant_prefix=BARRIER_PLACEHOLDER):
    if additional is None:
        additional = set()

    terms = load_terms(terms_file, labels) | additional
    target_col = 'abstract_lem'
    documents, titles = load_documents(preproc_file, target_col,
                                       barrier_placeholder, relevant_prefix)
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in terms]
        good_docs.append(gd)
    return good_docs, titles


def load_additional_terms(input_file):
    """
    Loads a list of keyword terms from a file

    This functions skips all the lines that starts with a '#'.
    Each term is split in a tuple of strings

    :param input_file: file to read
    :type input_file: str
    :return: the loaded terms as a set of strings
    :rtype: set[str]
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        rel_words_list = f.read().splitlines()

    rel_words_list = {w for w in rel_words_list if w != '' and w[0] != '#'}

    return rel_words_list


def prepare_documents(preproc_file, terms_file, ngrams, labels,
                      additional_keyword=None,
                      barrier_placeholder=BARRIER_PLACEHOLDER,
                      relevant_prefix=BARRIER_PLACEHOLDER):
    """
    Elaborates the documents preparing the bag of word representation

    :param preproc_file: path to the csv file with the lemmatized abstracts
    :type preproc_file: str
    :param terms_file: path to the csv file with the classified terms
    :type terms_file: str
    :param ngrams: if True use all the ngrams
    :type ngrams: bool
    :param labels: use only the terms classified with the labels specified here
    :type labels: tuple[str]
    :param additional_keyword: additional keyword loaded from file
    :type additional_keyword: set or None
    :param barrier_placeholder: placeholder for barrier words
    :type barrier_placeholder: str
    :param relevant_prefix: prefix used to mark relevant terms
    :type relevant_prefix: str
    :return: the documents as bag of words and the document titles
    :rtype: tuple[list[list[str]], list[str]]
    """
    if additional_keyword is None:
        additional_keyword = set()
    if ngrams:
        docs, titles = generate_filtered_docs_ngrams(terms_file, preproc_file,
                                                     labels, additional_keyword,
                                                     barrier_placeholder,
                                                     relevant_prefix)
    else:
        docs, titles = generate_filtered_docs(terms_file, preproc_file, labels,
                                              additional_keyword,
                                              barrier_placeholder,
                                              relevant_prefix)

    return docs, titles


def train_lda_model(docs, topics=20, alpha='auto', beta='auto', no_above=0.5,
                    no_below=20, seed=None):
    """
    Trains the lda model

    Each parameter has the default value used also as default by the lda.py script
    :param docs: documents train the model upon
    :type docs: list[list[str]]
    :param topics: number of topics
    :type topics: int
    :param alpha: alpha parameter
    :type alpha: float or str
    :param beta: beta parameter
    :type beta: float or str
    :param no_above: keep terms which are contained in no more than this
        fraction of documents (fraction of total corpus size, not an absolute
        number).
    :type no_above: float
    :param no_below: keep tokens which are contained in at least this number of
        documents (absolute number).
    :type no_below: int
    :param seed: seed used for random generator
    :type seed: int or None
    :return: the trained model and the dictionary object used in training
    :rtype: tuple[LdaModel, Dictionary]
    """
    # Make a index to word dictionary.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below,
                               no_above=no_above)
    _ = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # Train LDA model.
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=len(corpus),
        alpha=alpha,
        eta=beta,
        num_topics=topics,
        random_state=seed
    )
    return model, dictionary


def prepare_topics(model, docs, titles, dictionary):
    """
    Prepare the dicts for the topics and the document topic assignment

    :param model: the trained lda model
    :type model: LdaModel
    :param docs: the documents to evaluate to assign the topics
    :type docs: list[list[str]]
    :param titles: the titles of the documents
    :type titles: list[str]
    :param dictionary: the gensim dictionary object used for training
    :type dictionary: Dictionary
    :return: the dict of the topics, the docs-topics assignement and
        the average coherence score
    :rtype: tuple[dict[int, dict[str, str or dict[str, float]]],
        list[dict[str, int or str or dict[int, str]]], float]
    """
    cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary,
                        coherence='c_v', processes=PHYSICAL_CPUS)

    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    avg_topic_coherence = cm.get_coherence()
    coherence = cm.get_coherence_per_topic()
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

    return topics, docs_topics, avg_topic_coherence


def main():
    args = init_argparser().parse_args()

    terms_file = args.dataset / f'{args.prefix}_terms.csv'
    preproc_file = args.dataset / f'{args.prefix}_preproc.csv'

    barrier_placeholder = args.placeholder
    relevant_prefix = barrier_placeholder

    if args.no_relevant:
        labels = ('keyword',)
    else:
        labels = ('keyword', 'relevant')

    additional_keyword = set()

    if args.additional_file is not None:
        for sfile in args.additional_file:
            additional_keyword |= load_additional_terms(sfile)

    docs, titles = prepare_documents(preproc_file, terms_file,
                                     args.ngrams, labels,
                                     additional_keyword,
                                     barrier_placeholder,
                                     relevant_prefix)

    if args.load_model is not None:
        lda_path = Path(args.load_model)
        model = LdaModel.load(str(lda_path / 'model'))
        dictionary = Dictionary.load(str(lda_path / 'model_dictionary'))
    else:
        no_below = args.no_below
        no_above = args.no_above
        topics = args.topics
        alpha = args.alpha
        beta = args.beta
        seed = args.seed
        model, dictionary = train_lda_model(docs, topics, alpha, beta,
                                            no_above, no_below, seed)

    docs_topics, topics, avg_topic_coherence = prepare_topics(model, docs,
                                                              titles,
                                                              dictionary)

    print(f'Average topic coherence: {avg_topic_coherence:.4f}.')
    now = datetime.now()
    name = f'{args.prefix}_terms-topics_{now:%Y-%m-%d_%H%M%S}.json'
    topic_file = args.dataset / name
    with open(topic_file, 'w') as file:
        json.dump(topics, file, indent='\t')

    name = f'{args.prefix}_docs-topics_{now:%Y-%m-%d_%H%M%S}.json'
    docs_file = args.dataset / name
    with open(docs_file, 'w') as file:
        json.dump(docs_topics, file, indent='\t')

    if args.model:
        lda_path: Path = args.dataset / f'{args.prefix}_lda_model'
        lda_path.mkdir(exist_ok=True)
        model.save(str(lda_path / 'model'))
        dictionary.save(str(lda_path / 'model_dictionary'))


if __name__ == '__main__':
    main()
