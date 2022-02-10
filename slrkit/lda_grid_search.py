import logging
import pathlib
import sys
import uuid

# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

from datetime import datetime
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

from slrkit_utils.argument_parser import ArgParse, ValidateInt
from .lda import PHYSICAL_CPUS, prepare_documents, prepare_corpus, save_toml_files
from .utils import STOPWORD_PLACEHOLDER, setup_logger

# these globals are used by the multiprocess workers used in compute_optimal_model
_corpora: Optional[Dict[Tuple[str], Tuple[List[Tuple[int, int]],
                                          Dictionary, List[List[str]],
                                          int]]] = None
_seed: Optional[int] = None
_logger: Optional[logging.Logger] = None
_modeldir: Optional[pathlib.Path] = None


def to_ignore(_):
    return ['lda*.json', '*lda_results/']


def init_argparser():
    """Initialize the command line parser."""
    epilog = 'The script tests different lda models with different ' \
             'parameters and it tries to find the best model. The optimal ' \
             'number of topic is searched in the interval specified by the ' \
             'user on the command line.'
    parser = ArgParse(description='Performs the LDA on a dataset', epilog=epilog)
    parser.add_argument('preproc_file', action='store', type=Path,
                        help='path to the the preprocess file with the text to '
                             'elaborate.', input=True)
    parser.add_argument('terms_file', action='store', type=Path,
                        help='path to the file with the classified terms.',
                        input=True)
    parser.add_argument('outdir', action='store', type=Path, nargs='?',
                        default=Path.cwd(), help='path to the directory where '
                                                 'to save the results.')
    parser.add_argument('--text-column', '-t', action='store', type=str,
                        default='abstract_lem', dest='target_column',
                        help='Column in preproc_file to process. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--title-column', action='store', type=str,
                        default='title', dest='title',
                        help='Column in preproc_file to use as document title. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--min-topics', '-m', type=int,
                        default=5, action=ValidateInt,
                        help='Minimum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--max-topics', '-M', type=int,
                        default=20, action=ValidateInt,
                        help='Maximum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--step-topics', '-s', type=int,
                        default=1, action=ValidateInt,
                        help='Step in range(min,max,step) for topics retrieving'
                             ' (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed to be used in training. Default %(default)r')
    parser.add_argument('--plot-show', action='store_true',
                        help='if set, it plots the coherence')
    parser.add_argument('--plot-save', action='store_true',
                        help='if set, it saves the plot of the coherence as '
                             '<outdir>/lda_plot.pdf')
    parser.add_argument('--placeholder', '-p',
                        default=STOPWORD_PLACEHOLDER,
                        help='Placeholder for barrier word. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)s')
    parser.add_argument('--delimiter', action='store', type=str,
                        default='\t', help='Delimiter used in preproc_file. '
                                           'Default %(default)r')
    parser.add_argument('--logfile', default='slr-kit.log',
                        help='log file name. If omitted %(default)r is used',
                        logfile=True)
    return parser


def init_train(corpora, seed, modeldir, logger):
    global _corpora, _seed, _logger, _modeldir
    _corpora = corpora
    _seed = seed
    _logger = logger
    _modeldir = modeldir


# c_idx is the corpus index, n_topics is the number of topics,
# a is alpha and _b is beta
def train(c_idx, n_topics, _a, _b):
    global _corpora, _seed, _logger, _modeldir
    start = timer()
    corpus, dictionary, texts, n_docs = _corpora[c_idx]
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
    uid = str(uuid.uuid4())
    if isinstance(_a, float):
        alpha = '{:.3}'.format(_a)
    else:
        alpha = _a
    if isinstance(_b, float):
        beta = '{:.3}'.format(_b)
    else:
        beta = _b
    _logger.debug('{:.4f} {} {} {} {} {} {:.3f}'.format(c_v, uid, c_idx,
                                                        n_topics, alpha,
                                                        beta, stop - start))
    output_dir = _modeldir / uid
    output_dir.mkdir(exist_ok=True)
    model.save(str(output_dir / 'model'))
    dictionary.save(str(output_dir / 'model_dictionary'))
    return c_idx, n_topics, _a, _b, c_v, stop - start, uid, n_docs, len(corpus)


def compute_optimal_model(corpora, topics_range, alpha, beta, modeldir, logger,
                          seed=None):
    """
    Train several models iterating over the specified number of topics and performs
    LDA hyper-parameters alpha and beta tuning

    :param corpora: Gensim corpus
    :type corpora: dict[tuple[str, ...], tuple[list[tuple[int, int]], Dictionary, list[list[str]], int]]
    :param topics_range: range of the topics number to test
    :type topics_range: range
    :param alpha: Alpha parameter values to test. Every value accepted by gensim
    is valid.
    :type alpha: list[float or str]
    :param beta: Beta parameter values to test. Every value accepted by gensim
    is valid.
    :type beta: list[float or str]
    :param modeldir: directory where all the workers write the resulting model
    :type modeldir: pathlib.Path
    :param logger: logger object that all the workers will use
    :type logger: logging.Logger
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
        'uuid': [],
        'num_docs': [],
        'num_not_empty': [],
    }
    # iterate through all the combinations
    with Pool(processes=PHYSICAL_CPUS, initializer=init_train,
              initargs=(corpora, seed, modeldir, logger)) as pool:
        results = pool.starmap(train, product(corpora.keys(), topics_range,
                                              alpha, beta))
        # get the coherence score for the given parameters
        # LDA multi-core implementation with maximum number of workers
        for res in results:
            c, n, a, b, cv, t, uid, n_docs, n_not_empty = res
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
            model_results['uuid'].append(uid)
            model_results['num_docs'].append(n_docs)
            model_results['num_not_empty'].append(n_not_empty)

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


def lda_grid_search(args):
    logfile = args.logfile
    terms_file = args.terms_file
    preproc_file = args.preproc_file

    logger = setup_logger('debug_logger', logfile, level=logging.DEBUG)
    logger.info('==== lda_grid_search started ====')
    placeholder = args.placeholder
    relevant_prefix = placeholder

    if args.min_topics > args.max_topics:
        sys.exit('max_topics must be greater than min_topics')

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
    for labels in [('keyword', 'relevant'), ('keyword',)]:
        docs, titles = prepare_documents(preproc_file, terms_file,
                                         labels, args.target_column, args.title,
                                         delimiter=args.delimiter,
                                         placeholder=placeholder,
                                         relevant_prefix=relevant_prefix)

        tenth_of_titles = len(titles) // 10
        no_below_list_base = [1, 20, 40, 100, tenth_of_titles]
        no_below_list = [b for b in no_below_list_base if b <= tenth_of_titles]
        for no_below, no_above in product(no_below_list, no_above_list):
            corpus, dictionary = prepare_corpus(docs, no_above, no_below)
            if corpus is None:
                msg = (f'Combination {(labels, no_below, no_above)!r} skipped: '
                       f'all documents are empty')
                print(msg, file=sys.stderr)
                logger.info(msg)
            else:
                not_empty_bows = []
                not_empty_docs = []
                for c, d in zip(corpus, docs):
                    if c:
                        not_empty_bows.append(c)
                        not_empty_docs.append(d)

                n_docs = len(corpus)
                n_not_empty = len(not_empty_bows)
                empty_docs = n_docs - n_not_empty
                msg = f'Combination {(labels, no_below, no_above)!r} has ' \
                      f'{n_not_empty} non empty documents and {empty_docs} empty ' \
                      f'document'
                print(msg, file=sys.stderr)
                logger.info(msg)
                corpora[(labels, no_below, no_above)] = (not_empty_bows,
                                                         dictionary,
                                                         not_empty_docs,
                                                         n_docs)

    logger.info('N° of training process {}'.format(len(corpora)
                                                   * len(alpha)
                                                   * len(beta)
                                                   * len(topics_range)))
    # prepare result directories
    now = datetime.now()
    result_dir = args.outdir / f'{now:%Y-%m-%d_%H%M%S}_lda_results'
    result_dir.mkdir(exist_ok=True, parents=True)
    modeldir = result_dir / 'models'
    modeldir.mkdir(exist_ok=True, parents=True)

    results = compute_optimal_model(corpora, topics_range, alpha, beta,
                                    modeldir, logger, seed=args.seed)
    results.sort_values(by='coherence', ascending=False, inplace=True)
    results.reset_index(inplace=True, drop=True)
    results.to_csv(result_dir / 'results.csv', sep='\t', index_label='id')
    save_toml_files(args, results, result_dir)
    best = results.loc[results['coherence'].idxmax()]

    print('Best model:')
    print(best)

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
            fig_file = result_dir / 'lda_plot.pdf'
            plt.savefig(str(fig_file), dpi=1000)

    logger.info('==== lda_grid_search ended ====')


def main():
    args = init_argparser().parse_args()
    lda_grid_search(args)


if __name__ == '__main__':
    main()
