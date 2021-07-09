import logging
import pathlib
import sys

import random
import argparse
from datetime import datetime
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Tuple, Dict

from deap import base, creator, algorithms, tools
import numpy as np
import pandas as pd

# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings

    warnings.simplefilter('ignore')

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from slrkit_utils.argument_parser import AppendMultipleFilesAction, ArgParse
from lda import (PHYSICAL_CPUS, prepare_documents, load_acronyms,
                 prepare_corpus)
from utils import STOPWORD_PLACEHOLDER, setup_logger

# these globals are used by the multiprocess workers used in compute_optimal_model
_corpus: Optional[List[List[str]]] = None
_seed: Optional[int] = None
_logger: Optional[logging.Logger] = None


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
    parser = ArgParse(description='Performs the LDA on a dataset', epilog=epilog)
    parser.add_argument('preproc_file', action='store', type=Path,
                        help='path to the the preprocess file with the text to '
                             'elaborate.', input=True)
    parser.add_argument('terms_file', action='store', type=Path,
                        help='path to the file with the classified terms.',
                        input=True)
    parser.add_argument('outdir', action='store', type=Path, nargs='?',
                        default=Path.cwd(), help='path to the directory where '
                                                 'to save the results.',
                        non_standard=True)
    parser.add_argument('--text-column', '-t', action='store', type=str,
                        default='abstract_lem', dest='target_column',
                        help='Column in preproc_file to process. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--title-column', action='store', type=str,
                        default='title', dest='title',
                        help='Column in preproc_file to use as document title. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--no-ngrams', action='store_true',
                        help='if set do not use the ngrams')
    parser.add_argument('--additional-terms', '-T',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='additional_file',
                        help='Additional keywords files')
    parser.add_argument('--acronyms', '-a',
                        help='TSV files with the approved acronyms')
    parser.add_argument('--model', action='store_true',
                        help='if set, the best lda model is saved to directory '
                             '<outdir>/lda_model')
    parser.add_argument('--min-topics', '-m', type=int,
                        default=5, action=_ValidateInt,
                        help='Minimum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--max-topics', '-M', type=int,
                        default=20, action=_ValidateInt,
                        help='Maximum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--step-topics', '-s', type=int,
                        default=1, action=_ValidateInt,
                        help='Step in range(min,max,step) for topics retrieving'
                             ' (default: %(default)s)')
    parser.add_argument('--seed', type=int, action=_ValidateInt,
                        help='Seed to be used in training')
    parser.add_argument('--plot-show', action='store_true',
                        help='if set, it plots the coherence')
    parser.add_argument('--plot-save', action='store_true',
                        help='if set, it saves the plot of the coherence as '
                             '<outdir>/lda_plot.pdf')
    parser.add_argument('--result', '-r', metavar='FILENAME',
                        help='Where to save the training results '
                             'in CSV format. If "-", stdout is used.')
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


def init_train(corpora, seed, logger):
    global _corpus, _seed, _logger
    _corpus = corpora
    _seed = seed
    _logger = logger


def compute_optimal_model(corpora, topics_range, alpha, beta, outdir, logger,
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
    :param outdir: directory where all the workers write the resulting model
    :type outdir: pathlib.Path
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
              initargs=(corpora, seed, outdir, logger)) as pool:
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


# topics, alpha, beta, no_above, no_below label
def evaluate(ind):
    global _corpus, _seed, _logger
    # unpack parameter
    n_topics = ind[0]
    if ind[5] == 0:
        alpha = ind[1]
    elif ind[5] > 0:
        alpha = 'symmetric'
    else:
        alpha = 'asymmetric'

    beta = ind[2]
    no_above = ind[3]
    no_below = ind[4]
    start = timer()
    dictionary = Dictionary(_corpus)
    # Filter out words that occur less than no_above documents, or more than
    # no_below % of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    try:
        _ = dictionary[0]  # This is only to "load" the dictionary.
    except KeyError:
        return (-float('inf'),)

    not_empty_bows = []
    not_empty_docs = []
    for c in _corpus:
        bow = dictionary.doc2bow(c)
        if bow:
            not_empty_bows.append(bow)
            not_empty_docs.append(c)

    model = LdaModel(not_empty_bows, num_topics=n_topics,
                     id2word=dictionary, chunksize=len(not_empty_bows),
                     passes=10, random_state=_seed,
                     minimum_probability=0.0, alpha=alpha, eta=beta)
    # computes coherence score for that model
    cv_model = CoherenceModel(model=model, texts=not_empty_docs,
                              dictionary=dictionary, coherence='c_v',
                              processes=1)
    c_v = cv_model.get_coherence()
    stop = timer()
    return (c_v,)


def random_label():
    return random.choices([0, -1, 1],
                          [0.6, 0.2, 0.2], k=1)[0]


def check_bounds(val, min_, max_):
    if val > max_:
        return max_
    elif val < min_:
        return min_
    else:
        return val


# topics, alpha, beta, no_above, no_below label
def check_types(max_no_below, min_topics, max_topics):
    def decorator(func):
        def wrapper(*args, **kargs):
            inf = float('inf')
            offspring = func(*args, **kargs)
            for child in offspring:
                # topics
                if isinstance(child[0], float):
                    child[0] = int(np.round(child[0]))
                    child[0] = check_bounds(child[0], min_topics, max_topics)
                # no_below
                if isinstance(child[4], float):
                    child[4] = int(np.round(child[4]))
                    child[4] = check_bounds(child[4], 1, max_no_below)
                # no_above
                child[3] = check_bounds(child[3], 0.0, 1.0)
                # alpha and beta
                # alpha and beta have no max bound
                child[1] = check_bounds(child[1], 0.0, inf)
                child[2] = check_bounds(child[2], 0.0, inf)
                # other
                if isinstance(child[5], float):
                    ch = [-1, 0, 1]
                    v = int(np.floor(child[5]))
                    if v != 0:
                        v = np.sign(v)

                    ch.remove(v)
                    child[5] = random.choice(ch)

            return offspring

        return wrapper

    return decorator


def lda_grid_search(args):
    terms_file = args.terms_file
    preproc_file = args.preproc_file
    output_dir = args.outdir
    output_dir.mkdir(exist_ok=True)
    logfile = args.logfile
    if args.result is None:
        now = datetime.now()
        result_file = f'{now:%Y-%m-%d_%H%M%S}_results.csv'
    else:
        result_file = args.result

    logger = setup_logger('debug_logger', logfile, level=logging.DEBUG)
    logger.info('==== lda_ga_grid_search started ====')
    placeholder = args.placeholder
    relevant_prefix = placeholder

    if args.min_topics > args.max_topics:
        sys.exit('max_topics must be greater than min_topics')

    additional_keyword = set()

    if args.additional_file is not None:
        for sfile in args.additional_file:
            additional_keyword |= load_additional_terms(sfile)

    if args.acronyms is not None:
        acronyms = load_acronyms(args)
    else:
        acronyms = None

    docs, titles = prepare_documents(preproc_file, terms_file,
                                     not args.no_ngrams, ('keyword', 'relevant'),
                                     args.target_column, args.title,
                                     delimiter=args.delimiter,
                                     additional_keyword=additional_keyword,
                                     acronyms=acronyms,
                                     placeholder=placeholder,
                                     relevant_prefix=relevant_prefix)

    tenth_of_titles = len(titles) // 10

    # ga preparation
    if args.seed is not None:
        random.seed(args.seed)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('init_topics', random.randint, args.min_topics,
                     args.max_topics)
    toolbox.register('init_no_below', random.randint, 1, tenth_of_titles)
    toolbox.register('init_float', random.random)
    toolbox.register('init_label', random_label)
    # topics, alpha, beta, no_above, no_below, label
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.init_topics, toolbox.init_float,
                      toolbox.init_float, toolbox.init_float,
                      toolbox.init_no_below, toolbox.init_label), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
    toolbox.decorate('mate', check_types(tenth_of_titles, args.min_topics,
                                         args.max_topics))
    toolbox.decorate('mutate', check_types(tenth_of_titles, args.min_topics,
                                           args.max_topics))

    toolbox.register('select', tools.selTournament, tournsize=10)
    toolbox.register('evaluate', evaluate)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    pop = toolbox.population(n=102)
    with Pool(processes=6, initializer=init_train,
              initargs=(docs, args.seed, logger)) as pool:
        toolbox.register('map', pool.map)
        final_pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                                       mu=100, lambda_=24,
                                                       cxpb=0.5, mutpb=0.3,
                                                       ngen=4, halloffame=hof,
                                                       stats=stats,
                                                       verbose=True)
    print(hof)
    logger.info('==== lda_ga_grid_search ended ====')


def main():
    args = init_argparser().parse_args()
    lda_grid_search(args)


if __name__ == '__main__':
    main()
