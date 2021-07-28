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
from typing import Optional, List, Union, ClassVar

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


class BoundsNotSetError(Exception):
    pass


@dataclasses.dataclass
class LdaIndividual:
    """
    Represents an individual, a set of parameters for the LDA model

    The fitness attribute is used by DEAP for the optimization
    topics_bounds, max_no_below, min_no_above are class attribute and are used
    as bounds for the topics, no_above and no_below parameters.
    They must be set with the set_bounds class method before every operation,
    even the creation of a new instance.
    All the other attribute are protected. To access one value, use the
    corresponding property, or its index.
    The association between index and attribute is given by the order_from_name
    and name_from_order class methods.
    The alpha property is used to retrive the actual alpha value to use from the
    _alpha_val and _alpha_type values.
    The random_individual creates a new individual with random values.
    """
    _topics: int
    _alpha_val: float
    _beta: float
    _no_above: float
    _no_below: int
    _alpha_type: int
    fitness: creator.FitnessMax = dataclasses.field(init=False)
    topics_bounds: ClassVar[range] = None
    max_no_below: ClassVar[int] = None
    min_no_above: ClassVar[int] = None

    def __post_init__(self):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self.fitness = creator.FitnessMax()
        # these assignments triggers the checks on the values
        self.topics = self._topics
        self.alpha_val = self._alpha_val
        self.beta = self._beta
        self.no_above = self._no_above
        self.no_below = self._no_below
        self.alpha_type = self._alpha_type

    @classmethod
    def set_bounds(cls, min_topics, max_topics, max_no_below, min_no_above):
        """
        Sets the bounds used in properties to check the values

        Must be called before every operation
        :param min_topics: minimum number of topics
        :type min_topics: int
        :param max_topics: maximum number of topics
        :type max_topics: int
        :param max_no_below: maximum value for the no_below parameter
        :type max_no_below: int
        :param min_no_above: minimum value for the no_above parameter
        :type min_no_above: float
        :raise ValueError: if min_no_above is > 1.0
        """
        cls.topics_bounds = range(min_topics, max_topics + 1)
        cls.max_no_below = max_no_below
        if min_no_above > 1.0:
            raise ValueError('min_no_above must be less then 1.0')
        cls.min_no_above = min_no_above

    @classmethod
    def index_from_name(cls, name: str) -> int:
        """
        Gives the index of a parameter given its name

        :param name: name of the parameter
        :type name: str
        :return: the index of the parameter
        :rtype: int
        :raise ValueError: if the name is not valid
        """
        for i, f in enumerate(dataclasses.fields(cls)):
            if f.name == '_' + name:
                return i
        else:
            raise ValueError(f'{name!r} is not a valid field name')

    @classmethod
    def name_from_index(cls, index: int) -> str:
        """
        Gives the name of a parameter given its index

        :param index: index of the parameter
        :type index: str
        :return: the name of the parameter
        :rtype: str
        :raise ValueError: if the index is not valid
        """
        try:
            name = dataclasses.fields(cls)[index].name
        except IndexError:
            raise ValueError(f'{index!r} is not a valid index')
        return name.strip('_')

    @classmethod
    def random_individual(cls, prob_no_filters=0.5):
        """
        Creates a random individual

        The prob_no_filters is the probability that the created individual has
        no_below == no_above == 1.
        :param prob_no_filters: probability that the individual has no_below
            and no_above set to 1
        :type prob_no_filters: float
        :return: the new individual
        :rtype: LdaIndividual
        :raise BoundsNotSetError: if the set_bounds method is not called first
        """
        if cls.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')

        no_below = 1
        no_above = 1.0
        if random.random() < 1 - prob_no_filters:
            no_below = random.randint(1, cls.max_no_below)
            no_above = random.uniform(cls.min_no_above, 1.0)

        return LdaIndividual(_topics=random.randint(cls.topics_bounds.start,
                                                    cls.topics_bounds.stop),
                             _alpha_val=random.random(),
                             _beta=random.random(),
                             _no_above=no_above,
                             _no_below=no_below,
                             _alpha_type=random.choices([0, 1, -1],
                                                        [0.6, 0.2, 0.2],
                                                        k=1)[0])

    @property
    def topics(self):
        return self._topics

    @topics.setter
    def topics(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._topics = check_bounds(int(np.round(val)),
                                    self.topics_bounds.start,
                                    self.topics_bounds.stop)

    @property
    def alpha_val(self):
        return self._alpha_val

    @alpha_val.setter
    def alpha_val(self, val):
        self._alpha_val = check_bounds(val, 0.0, float('inf'))

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = check_bounds(val, 0.0, float('inf'))

    @property
    def no_above(self):
        return self._no_above

    @no_above.setter
    def no_above(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._no_above = check_bounds(val, self.min_no_above, 1.0)

    @property
    def no_below(self):
        return self._no_below

    @no_below.setter
    def no_below(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._no_below = check_bounds(int(np.round(val)), 1, self.max_no_below)

    @property
    def alpha_type(self):
        return self._alpha_type

    @alpha_type.setter
    def alpha_type(self, val):
        v = int(np.round(val))
        if v != 0:
            v = np.sign(v)
        self._alpha_type = v

    @property
    def alpha(self) -> Union[float, str]:
        if self._alpha_type == 0:
            return self._alpha_val
        elif self._alpha_type > 0:
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
            name = self.name_from_index(i)
            setattr(self, name, value[val_index])


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
    parser.add_argument('ga_params', action='store', type=Path,
                        help='path to the file with the parameters for the ga.')
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


def load_ga_params(args):
    with open(args.ga_params) as file:
        params = tomlkit.loads(file.read())
    default_params_file = Path(__file__).parent / 'ga_param.toml'
    with open(default_params_file) as file:
        defaults = tomlkit.loads(file.read())
    for sec in defaults.keys():
        if sec not in params:
            params.add(sec, defaults[sec])
            continue
        for k, v in defaults[sec].items():
            if k not in params[sec]:
                params[sec].add(k, v)
            elif sec == 'mutate':
                if 'mu' not in params[sec][k]:
                    params[sec][k].add('mu', v['mu'])
                if 'sigma' not in params[sec][k]:
                    params[sec][k].add('sigma', v['sigma'])

    del file, defaults, default_params_file
    return params


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
    params = load_ga_params(args)
    if params['limits']['min_topics'] > params['limits']['max_topics']:
        sys.exit('limits.max_topics must be greater than limits.min_topics')

    if (params['probabilities']['no_filter'] < 0
            or params['probabilities']['no_filter'] > 1.0):
        sys.exit('probabilities.no_filter must be a value between 0 and 1')

    if params['probabilities']['mate'] + params['probabilities']['mutate'] > 1:
        sys.exit('The sum of the crossover and mutation probabilities must be '
                 'smaller or equal to 1.0')

    # ga preparation
    if args.seed is not None:
        random.seed(args.seed)

    # set the bound used by LdaIndividual to check the topics and no_below values
    max_no_below = params['limits']['max_no_below']
    if max_no_below == -1:
        max_no_below = tenth_of_titles
    elif max_no_below >= len(titles):
        sys.exit('max_no_below cannot have the same value as the number of documents')

    try:
        LdaIndividual.set_bounds(params['limits']['min_topics'],
                                 params['limits']['max_topics'],
                                 max_no_below, params['limits']['min_no_above'])
    except ValueError as e:
        sys.exit(e.args[0])

    creator.create('Individual', LdaIndividual, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', LdaIndividual.random_individual,
                     prob_no_filters=params['probabilities']['no_filter'])
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxTwoPoint)
    mut_mu = [0.0] * len(params['mutate'])
    mut_sigma = list(mut_mu)
    for f, v in params['mutate'].items():
        i = LdaIndividual.index_from_name(f)
        mut_mu[i] = float(v['mu'])
        mut_sigma[i] = float(v['sigma'])

    toolbox.register('mutate', tools.mutGaussian, mu=mut_mu, sigma=mut_sigma,
                     indpb=params['probabilities']['component_mutation'])
    toolbox.register('select', tools.selTournament,
                     tournsize=params['algorithm']['tournament_size'])
    toolbox.register('evaluate', evaluate)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    pop = toolbox.population(n=params['algorithm']['initial'])
    mu = params['algorithm']['mu']
    lambda_ = params['algorithm']['lambda']
    ngen = params['algorithm']['generations']
    estimated_trainings = len(pop) + lambda_ * ngen
    print('Estimated trainings:', estimated_trainings)
    logger.info(f'Estimated trainings: {estimated_trainings}')
    q = Queue(estimated_trainings)
    with Pool(processes=PHYSICAL_CPUS, initializer=init_train,
              initargs=(docs, args.seed, logger, q, output_dir)) as pool:
        toolbox.register('map', pool.map)
        _, _ = algorithms.eaMuPlusLambda(pop, toolbox,
                                         mu=mu, lambda_=lambda_,
                                         cxpb=params['probabilities']['mate'],
                                         mutpb=params['probabilities']['mutate'],
                                         ngen=ngen, stats=stats, verbose=True)
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
