import argparse
import dataclasses
import logging
import pathlib
import random
import sys
import uuid
from datetime import datetime
from multiprocessing import Pool, Queue
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import tomlkit
from deap import base, creator, algorithms, tools

# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings

    warnings.simplefilter('ignore')

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from slrkit_utils.argument_parser import AppendMultipleFilesAction, ArgParse
from lda import (PHYSICAL_CPUS, prepare_documents, load_acronyms)
from utils import STOPWORD_PLACEHOLDER, setup_logger

# these globals are used by the multiprocess workers used in compute_optimal_model
_corpus: Optional[List[List[str]]] = None
_seed: Optional[int] = None
_logger: Optional[logging.Logger] = None
_queue: Optional[Queue] = None
_outdir: Optional[pathlib.Path] = None


class _ValidateInt(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            v = int(values)
        except ValueError:
            parser.exit(1, f'Value {values!r} is not a valid int')
            return

        if v <= 0:
            parser.exit(1, f'{self.dest!r} must be greater than 0')

        setattr(namespace, self.dest, v)


class _ValidateProb(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            v = float(values)
        except ValueError:
            parser.exit(1, f'Value {values!r} is not a valid float')
            return

        if v < 0:
            parser.exit(1, f'{self.dest!r} must be greater than 0')
        elif v > 1.0:
            parser.exit(1, f'{self.dest!r} must be less than 1')

        setattr(namespace, self.dest, v)


creator.create('FitnessMax', base.Fitness, weights=(1.0,))


@dataclasses.dataclass
class LdaIndividual:
    topics: int
    alpha_val: float
    beta: float
    no_above: float
    no_below: int
    alpha_type: int
    fitness: creator.FitnessMax = dataclasses.field(init=False)

    def __post_init__(self):
        self.fitness = creator.FitnessMax()

    @classmethod
    def random_individual(cls, min_topics, max_topics, max_no_below):
        return LdaIndividual(random.randint(min_topics, max_topics),
                             random.random(),
                             random.random(),
                             random.random(),
                             random.randint(1, max_no_below),
                             random.choices([0, 1, -1], [0.6, 0.2, 0.2], k=1)[0])

    @property
    def alpha(self) -> Union[float, str]:
        if self.alpha_type == 0:
            return self.alpha_val
        elif self.alpha_type > 0:
            return 'symmetric'
        else:
            return 'asymmetric'

    def __len__(self):
        return 6

    def __getitem__(self, item):
        return dataclasses.astuple(self, tuple_factory=list)[item]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            indexes = range(*key.indices(len(self)))
        elif isinstance(key, int):
            indexes = [key]
        else:
            msg = f'Invalid type for an index: {type(key).__name__!r}'
            raise TypeError(msg)

        if isinstance(value, (int, float)):
            value = [value]
        elif not isinstance(value, (tuple, list)):
            msg = f'Unsupported type for assignement: {type(value).__name__!r}'
            raise TypeError(msg)

        for val_index, i in enumerate(indexes):
            # Where is the switch statement when is needed?
            if i == 0:
                self.topics = int(np.round(value[val_index]))
            elif i in [1, 2, 3]:
                if not isinstance(value[val_index], float):
                    raise TypeError(f'Required float for assignement to {i} index')

                inf = float('inf')
                if i == 1:
                    self.alpha_val = check_bounds(value[val_index], 0.0, inf)
                elif i == 2:
                    self.beta = check_bounds(value[val_index], 0.0, inf)
                else:
                    self.no_above = check_bounds(value[val_index], 0.0, 1.0)
            elif i == 4:
                self.no_below = int(np.round(value[val_index]))
            elif i == 5:
                v = int(np.round(value[val_index]))
                if v != 0:
                    v = np.sign(v)
                self.alpha_type = v


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
    parser.add_argument('--min-topics', '-m', type=int,
                        default=5, action=_ValidateInt,
                        help='Minimum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--max-topics', '-M', type=int,
                        default=20, action=_ValidateInt,
                        help='Maximum number of topics to retrieve '
                             '(default: %(default)s)')
    parser.add_argument('--initial-population', '-P', type=int,
                        default=100, action=_ValidateInt,
                        help='Size of the initial population. (default: '
                             '%(default)s)')
    parser.add_argument('--mu', type=int, default=100, action=_ValidateInt,
                        help='Number of individuals selected for each new '
                             'generation. (default: %(default)s)')
    parser.add_argument('--lambda', type=int, default=20, action=_ValidateInt,
                        help='Number of individuals generated at each new '
                             'generation. (default: %(default)s)')
    parser.add_argument('--generations', '-G', type=int, default=20,
                        action=_ValidateInt,
                        help='Number of generations. (default: %(default)s)')
    parser.add_argument('--crossover', type=float, default=0.5,
                        action=_ValidateProb,
                        help='Crossover probability of the GA. '
                             '(default: %(default)s)')
    parser.add_argument('--mutation', type=float, default=0.2,
                        action=_ValidateProb,
                        help='Mutation probability of the GA. '
                             '(default: %(default)s)')
    parser.add_argument('--best-individuals', '-B', type=int, default=5,
                        action=_ValidateInt,
                        help='Number of best individuals to collect. '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, action=_ValidateInt,
                        help='Seed to be used in training. The ga uses seed + 1')
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


def init_train(corpora, seed, logger, queue, outdir):
    global _corpus, _seed, _logger, _queue, _outdir
    _corpus = corpora
    _seed = seed
    _logger = logger
    _queue = queue
    _outdir = outdir


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
def evaluate(ind: LdaIndividual):
    global _corpus, _seed, _logger, _queue, _outdir
    # unpack parameter
    n_topics = ind.topics
    alpha = ind.alpha
    beta = ind.beta
    no_above = ind.no_above
    no_below = ind.no_below
    result = {}
    u = str(uuid.uuid4())
    result['topics'] = n_topics
    result['alpha'] = alpha
    result['beta'] = beta
    result['no_above'] = no_above
    result['no_below'] = no_below
    result['uuid'] = u
    result['seed'] = _seed
    start = timer()
    dictionary = Dictionary(_corpus)
    # Filter out words that occur less than no_above documents, or more than
    # no_below % of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    try:
        _ = dictionary[0]  # This is only to "load" the dictionary.
    except KeyError:
        c_v = -float('inf')
        result['coherence'] = c_v
        result['time'] = 0
        _queue.put(result)
        return (c_v,)

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
    result['coherence'] = c_v
    result['time'] = stop - start
    _queue.put(result)
    output_dir = _outdir / u
    output_dir.mkdir(exist_ok=True)
    model.save(str(output_dir / 'model'))
    dictionary.save(str(output_dir / 'model_dictionary'))
    return (c_v,)


def check_bounds(val, min_, max_):
    if val > max_:
        return max_
    elif val < min_:
        return min_
    else:
        return val


# [topics, alpha, beta, no_above, no_below, label]
def check_types(max_no_below, min_topics, max_topics):
    def decorator(func):
        def wrapper(*args, **kargs):
            inf = float('inf')
            offspring = func(*args, **kargs)
            for child in offspring:
                # topics
                if isinstance(child[0], float):
                    child.topics = int(np.round(child[0]))
                    child.topics = check_bounds(child[0], min_topics, max_topics)
                # no_below
                if isinstance(child[4], float):
                    child.no_below = int(np.round(child[4]))
                    child.no_below = check_bounds(child[4], 1, max_no_below)

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
    # if args.seed is not None:
    #     random.seed(args.seed)
    creator.create('Individual', LdaIndividual, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', LdaIndividual.random_individual,
                     args.min_topics, args.max_topics, tenth_of_titles)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
    toolbox.decorate('mate', check_types(tenth_of_titles, args.min_topics,
                                         args.max_topics))
    toolbox.decorate('mutate', check_types(tenth_of_titles, args.min_topics,
                                           args.max_topics))

    toolbox.register('select', tools.selTournament, tournsize=5)
    toolbox.register('evaluate', evaluate)
    hof = tools.HallOfFame(args.best_individuals)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    pop = toolbox.population(n=args.initial_population)
    lambda_ = getattr(args, 'lambda')
    estimated_trainings = args.initial_population + lambda_ * args.generations
    print('Estimated trainings:', estimated_trainings)
    logger.info(f'Estimated trainings: {estimated_trainings}')
    q = Queue(estimated_trainings)
    with Pool(processes=PHYSICAL_CPUS, initializer=init_train,
              initargs=(docs, args.seed, logger, q, output_dir)) as pool:
        toolbox.register('map', pool.map)
        final_pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                                       mu=args.mu,
                                                       lambda_=lambda_,
                                                       cxpb=args.crossover,
                                                       mutpb=args.mutation,
                                                       ngen=args.generations,
                                                       halloffame=hof,
                                                       stats=stats,
                                                       verbose=True)
    results = []
    while not q.empty():
        results.append(q.get())
        u = results[-1]['uuid']
        conf = tomlkit.document()
        conf.add('preproc_file', str(preproc_file))
        conf.add('terms_file', str(terms_file))
        conf.add('outdir', str(output_dir))
        conf.add('text-column', args.target_column)
        conf.add('title-column', args.title)
        if args.additional_file is not None:
            conf.add('additional-terms', args.additional_file)
        else:
            conf.add('additional-terms', [])
        if args.acronyms is not None:
            conf.add('acronyms', args.acronyms)
        else:
            conf.add('acronyms', '')
        conf.add('topics', results[-1]['topics'])
        conf.add('alpha', results[-1]['alpha'])
        conf.add('beta', results[-1]['beta'])
        conf.add('no_below', results[-1]['no_below'])
        conf.add('no_above', results[-1]['no_above'])
        conf.add('seed', results[-1]['seed'])
        conf.add('no-ngrams', args.no_ngrams)
        conf.add('model', False)
        conf.add('no-relevant', False)
        conf.add('load-model', str(output_dir / u))
        conf.add('placeholder', placeholder)
        conf.add('delimiter', args.delimiter)
        with open(output_dir / ''.join([u, '.toml']), 'w') as file:
            file.write(tomlkit.dumps(conf))

    q.close()
    df = pd.DataFrame(results)
    df.sort_values(by=['coherence'], ascending=False, inplace=True)
    df.to_csv(result_file, sep='\t', index=False)
    with pd.option_context('display.width', 80,
                           'display.float_format', '{:,.3f}'.format):
        print(df)
    logger.info('==== lda_ga_grid_search ended ====')


def main():
    args = init_argparser().parse_args()
    lda_grid_search(args)


if __name__ == '__main__':
    main()
