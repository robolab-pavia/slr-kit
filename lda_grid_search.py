import argparse
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from lda import (PHYSICAL_CPUS, BARRIER_PLACEHOLDER, prepare_documents,
                 output_topics)

# these globals are used by the multiprocess workers used in compute_optimal_model
from utils import AppendMultipleFilesAction, assert_column

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
    epilog = 'The script tests different lda models with different ' \
             'parameters and it tries to find the best model. The optimal ' \
             'number of topic is searched in the interval specified by the ' \
             'user on the command line.'
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog=epilog)
    parser.add_argument('preproc_file', action='store', type=Path,
                        help='path to the the preprocess file with the text to '
                             'elaborate.')
    parser.add_argument('terms_file', action='store', type=Path,
                        help='path to the file with the classified terms.')
    parser.add_argument('outdir', action='store', type=Path, nargs='?',
                        default=Path.cwd(),
                        help='path to the directory where to save the results.')
    parser.add_argument('--text-column', '-t', action='store', type=str,
                        default='abstract_lem', dest='target_column',
                        help='Column in preproc_file to process. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--ngrams', action='store_true',
                        help='if set use all the ngrams')
    parser.add_argument('--additional-terms', '-T',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='additional_file',
                        help='Additional keywords files')
    parser.add_argument('--acronyms', '-a',
                        help='TSV files with the approved acronyms')
    parser.add_argument('--model', action='store_true',
                        help='if set, the best lda model is saved to directory '
                             '<outdir>/lda_model')
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
                             '<outdir>/lda_plot.pdf')
    parser.add_argument('--result', '-r', metavar='FILENAME',
                        type=argparse.FileType('w'), default='-',
                        help='Where to save the training results in CSV format.'
                             ' If omitted or -, stdout is used.')
    parser.add_argument('--output', '-o', action='store_true',
                        help='if set, it stores the topic description in '
                             '<outdir>/lda_terms-topics_<date>_<time>.json, '
                             'and the document topic assignment in '
                             '<outdir>/lda_docs-topics_<date>_<time>.json')
    parser.add_argument('--placeholder', '-p', default=BARRIER_PLACEHOLDER,
                        help='Placeholder for barrier word. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)s')
    return parser


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
    # iterate through all the combinations

    with Pool(processes=PHYSICAL_CPUS, initializer=init_train,
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


def main():
    args = init_argparser().parse_args()
    terms_file = args.terms_file
    preproc_file = args.preproc_file
    output_dir = args.outdir

    barrier_placeholder = args.placeholder
    relevant_prefix = barrier_placeholder

    if args.min_topics >= args.max_topics:
        sys.exit('max_topics must be greater than min_topics')

    additional_keyword = set()

    if args.additional_file is not None:
        for sfile in args.additional_file:
            additional_keyword |= load_additional_terms(sfile)

    if args.acronyms is not None:
        conv = {'Acronym': lambda s: s.lower()}
        acronyms = pd.read_csv(args.acronyms, delimiter='\t', encoding='utf-8',
                               converters=conv, usecols=list(conv))
        assert_column(args.acronyms, acronyms, ['Acronym'])
    else:
        acronyms = None

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
    no_above_list = [0.5, 0.6, 0.75, 1.0]
    for labels in [('keyword', 'relevant'), ('keyword', )]:
        docs, titles = prepare_documents(preproc_file, terms_file,
                                         args.ngrams, labels, args.target_column,
                                         additional_keyword=additional_keyword,
                                         acronyms=acronyms,
                                         barrier_placeholder=barrier_placeholder,
                                         relevant_prefix=relevant_prefix)

        tenth_of_titles = len(titles) // 10
        no_below_list_base = [1, 20, 40, 100, tenth_of_titles]
        no_below_list = [b for b in no_below_list_base if b <= tenth_of_titles]
        for no_below, no_above in product(no_below_list, no_above_list):
            corpus, dictionary = prepare_corpus(docs, no_above, no_below)
            corpora[(labels, no_below, no_above)] = (corpus, dictionary, docs)

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
            output_topics(model, dictionary, docs, titles, output_dir, 'lda')

        if args.model:
            lda_path = output_dir / 'lda_model'
            lda_path.mkdir(exist_ok=True)
            model.save(str(lda_path / 'model'))
            dictionary.save(str(lda_path / 'model_dictionary'))

    if args.plot_show or args.plot_save:
        max_cv = results.groupby('topics')['coherence'].idxmax()
        plt.plot(results.loc[max_cv, 'topics'],
                 results.loc[max_cv, 'coherence'],
                 marker='o', linestyle='solid')

        plt.xticks(topics_range)
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence score')
        plt.grid()
        if args.plot_show:
            plt.show()

        if args.plot_save:
            fig_file = output_dir / 'lda_plot.pdf'
            plt.savefig(str(fig_file), dpi=1000)


if __name__ == '__main__':
    main()
