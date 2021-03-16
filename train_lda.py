import argparse
import json
import sys
from itertools import repeat, product
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from psutil import cpu_count

from utils import substring_index

PHYSICAL_CPU = cpu_count(logical=False)

# these globals are used by the multiprocess workers used in compute_optimal_model
_corpora: Optional[Dict[Tuple[str], Tuple[List[Tuple[int, int]],
                                          Dictionary, List[List[str]]]]] = None
_seed: Optional[int] = None


class _ValidateInt(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if int(values) <= 0:
            parser.exit(1, f'{self.dest!r} must be greater than 0')

        setattr(namespace, self.dest, self.type(values))


def init_argparser():
    """Initialize the command line parser."""
    epilog = 'The program uses two files: <dataset>/<prefix>_preproc.csv and ' \
             '<dataset>/<prefix>_terms.csv. The first one is used for the ' \
             'documents abstract and title. The second one is used for the ' \
             'terms.\nThe script tests different lda models with different ' \
             'parameters and it tries to find the best model. The optimal ' \
             'number of topic is searched in the interval specified by the ' \
             'user on the command line.'
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog=epilog)
    parser.add_argument('dataset', action='store', type=Path,
                        help='path to the directory where the files of the '
                             'dataset to elaborate are stored.')
    parser.add_argument('prefix', action='store', type=str,
                        help='prefix used when searching files.')
    parser.add_argument('--ngrams', action='store_true',
                        help='if set use all the ngrams')
    parser.add_argument('--model', action='store_true',
                        help='if set, the best lda model is saved to directory '
                             '<dataset>/<prefix>_lda_model')
    parser.add_argument('--min-topics', '-m', type=int, default=5,
                        action=_ValidateInt,
                        help='Minimum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--max-topics', '-M', type=int, default=20,
                        action=_ValidateInt,
                        help='Maximum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--step-topics', '-s', type=int, default=1,
                        action=_ValidateInt,
                        help='Step in range(min,max,step) for topics retrieving '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, action=_ValidateInt,
                        help='Seed to be used in training')
    parser.add_argument('--plot-show', action='store_true',
                        help='if set, it plots the coherence')
    parser.add_argument('--plot-save', action='store_true',
                        help='if set, it saves the plot of the coherence as '
                             '<dataset>/<prefix>_lda_plot.pdf')
    parser.add_argument('--keyword-only', action='store_true',
                        help='if set, it uses only the keyword term')
    parser.add_argument('--result', '-r', metavar='FILENAME',
                        type=argparse.FileType('w'), default='-',
                        help='Where to save the training results in CSV format.'
                             ' If omitted or -, stdout is used.')
    parser.add_argument('--output', '-o', action='store_true',
                        help='if set, it stores the topic description in '
                             '<dataset>/<prefix>_topics.json, and the document '
                             'topic assignment in '
                             '<dataset>/<prefix>_docs-topics.json')
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


def generate_filtered_docs_ngrams(terms_file, preproc_file):
    labels = [('keyword', 'relevant'), ('keyword',)]
    docs = {}
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    titles = dataset['title'].to_list()
    for lbl in labels:
        terms = load_ngrams(terms_file, lbl)
        if all(len(t) == 0 for t in terms.values()):
            continue

        ngram_len = sorted(terms, reverse=True)
        documents = dataset['abstract_lem'].to_list()
        with Pool() as pool:
            docs[lbl] = pool.starmap(filter_doc, zip(documents,
                                                     repeat(ngram_len),
                                                     repeat(terms)))
    return docs, titles


def generate_filtered_docs(terms_file, preproc_file):
    labels = [('keyword', 'relevant'), ('keyword',)]
    good_docs = {}
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    titles = dataset['title'].to_list()
    for lbl in labels:
        terms = load_terms(terms_file, lbl)
        if len(terms) == 0:
            continue

        documents = dataset['abstract_lem'].to_list()
        docs = [d.split(' ') for d in documents]

        gd = []
        for doc in docs:
            gd.append([t for t in doc if t in terms])

        good_docs[lbl] = gd

    return good_docs, titles


def prepare_corpus(docs, no_above, no_below):
    dictionary = Dictionary(docs)
    # Filter out words that occur less than no_above documents, or more than
    # no_below % of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # Finally, we transform the documents to a vectorized form. We simply
    # compute the frequency of each word, including the bigrams.
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    return corpus, dictionary


def init_train(corpora, seed):
    global _corpora, _seed
    _corpora = corpora
    _seed = seed


# c_idx is the corpus index, n_topics is the number of topics,
# a is alpha and _b is beta
def train(c_idx, n_topics, _a, _b):
    global _corpora, _seed
    start = timer()
    corpus, dictionary, texts = _corpora[c_idx]
    model = LdaModel(corpus, num_topics=n_topics,
                     id2word=dictionary, chunksize=len(corpus),
                     passes=10, random_state=_seed,
                     minimum_probability=0.0, alpha=_a, eta=_b)
    # computes coherence score for that model
    cv_model = CoherenceModel(model=model, texts=texts,
                              dictionary=dictionary, coherence='c_v',
                              processes=1)
    c_v = cv_model.get_coherence()
    stop = timer()
    return c_idx, n_topics, _a, _b, c_v, stop - start


def compute_optimal_model(corpora, topics_range, alpha, beta, seed=None):
    """
    Train several models iterating over the specified number of topics and performs
    LDA hyper-parameters alpha and beta tuning

    :param corpora: Gensim corpus
    :type corpora: dict[tuple[str, ...], tuple[list[tuple[int, int]], Dictionary, list[list[str]]]]
    :param topics_range: range of the topics number to test
    :type topics_range: range
    :param alpha: Alpha parameter values to test. Every value accepted by gensim
    is valid.
    :type alpha: list[float or str]
    :param beta: Beta parameter values to test. Every value accepted by gensim
    is valid.
    :type beta: list[float or str]
    :param seed: random number generator seed
    :type seed: int or None
    :return: Dataframe with the model performances
    :rtype: pd.DataFrame
    """
    model_results = {
        'corpus': [],
        'no_below': [],
        'no_above': [],
        'topics': [],
        'alpha': [],
        'beta': [],
        'coherence': [],
        'times': [],
        'seed': [],
    }
    # Can take a long time to run
    corpora_len = len(corpora)

    # iterate through all the combinations

    with Pool(processes=PHYSICAL_CPU, initializer=init_train,
              initargs=(corpora, seed)) as pool:
        results = pool.starmap(train, product(corpora.keys(), topics_range,
                                              alpha, beta))
        # get the coherence score for the given parameters
        # LDA multi-core implementation with maximum number of workers
        for res in results:
            c, n, a, b, cv, t = res
            # Save the model results
            model_results['corpus'].append(c[0])
            model_results['no_below'].append(c[1])
            model_results['no_above'].append(c[2])
            model_results['topics'].append(n)
            model_results['alpha'].append(a)
            model_results['beta'].append(b)
            model_results['coherence'].append(cv)
            model_results['times'].append(t)
            model_results['seed'].append(seed)

    return pd.DataFrame(model_results)


def output_topics(model, docs, titles, args):
    cm = CoherenceModel(model=model, texts=docs, dictionary=model.id2word,
                        coherence='c_v', processes=PHYSICAL_CPU)
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
    topic_file = args.dataset / f'{args.prefix}_terms-topics.json'
    with open(topic_file, 'w') as file:
        json.dump(topics, file, indent='\t')
    docs_topics = []
    for i, (title, d) in enumerate(zip(titles, docs)):
        bow = model.id2word.doc2bow(d)
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


def main():
    args = init_argparser().parse_args()
    terms_file = args.dataset / f'{args.prefix}_terms.csv'
    preproc_file = args.dataset / f'{args.prefix}_preproc.csv'

    if args.min_topics >= args.max_topics:
        sys.exit('max_topics must be greater than min_topics')

    if args.ngrams:
        docs, titles = generate_filtered_docs_ngrams(terms_file, preproc_file)
    else:
        docs, titles = generate_filtered_docs(terms_file, preproc_file)

    no_below_list = [1, 20, 40, 100, len(titles)//10]
    no_above_list = [0.5, 0.6, 0.75, 1.0]
    topics_range = range(args.min_topics, args.max_topics + args.step_topics,
                         args.step_topics)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.1))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.1))
    beta.append('auto')
    corpora = {}
    for k, d in docs.items():
        for no_below, no_above in product(no_below_list, no_above_list):
            corpus, dictionary = prepare_corpus(d, no_above, no_below)
            corpora[(k, no_below, no_above)] = (corpus, dictionary, d)

    results = compute_optimal_model(corpora, topics_range, alpha, beta,
                                    args.seed)

    results.to_csv(args.result, index=False)
    best = results.loc[results['coherence'].idxmax()]

    print('Best model:')
    print(best)

    if args.output or args.model:
        corpus, dictionary, docs = corpora[(best['corpus'],
                                            best['no_below'],
                                            best['no_above'])]
        model = LdaModel(corpus, num_topics=best['topics'],
                         id2word=dictionary, chunksize=len(corpus),
                         passes=10, random_state=best['seed'],
                         minimum_probability=0.0,
                         alpha=best['alpha'], eta=best['beta'])

        if args.output:
            output_topics(model, docs, titles, args)

        if args.model:
            lda_path: Path = args.dataset / f'{args.prefix}_lda_model'
            lda_path.mkdir(exist_ok=True)
            model.save(str(lda_path / 'model'))
            dictionary.save(str(lda_path / 'model_dictionary'))

    if args.plot_show or args.plot_save:
        max_cv = results.groupby('topics')['coherence'].idxmax()
        plt.plot(results.loc[max_cv, 'topics'], results.loc[max_cv, 'coherence'],
                 marker='o', linestyle='solid')

        plt.xticks(topics_range)
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence score')
        plt.grid()
        if args.plot_show:
            plt.show()

        if args.plot_save:
            fig_file = args.dataset / f'{args.prefix}_lda_plot.pdf'
            plt.savefig(str(fig_file), dpi=1000)


if __name__ == '__main__':
    main()
