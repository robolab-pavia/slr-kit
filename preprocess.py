import abc
import logging
import re
import sys
from itertools import repeat
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import Generator, Tuple, Sequence

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from psutil import cpu_count

from arguments import (AppendMultipleFilesAction, AppendMultiplePairsAction,
                       ArgParse)
from utils import (setup_logger, assert_column,
                   log_end, log_start, STOPWORD_PLACEHOLDER, RELEVANT_PREFIX)

PHYSICAL_CPUS = cpu_count(logical=False)


class Lemmatizer(abc.ABC):
    @abc.abstractmethod
    def lemmatize(self, text: Sequence[str]) -> Generator[Tuple[str, str], None, None]:
        pass


class EnglishLemmatizer(Lemmatizer):
    def __init__(self):
        self._lem = WordNetLemmatizer()

    def lemmatize(self, text: Sequence[str]) -> Generator[Tuple[str, str], None, None]:
        for word in text:
            yield (word, self._lem.lemmatize(word))


class ItalianLemmatizer(Lemmatizer):
    def __init__(self, treetagger_dir=None):
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                import treetaggerwrapper as ttagger
        except ModuleNotFoundError:
            sys.exit('Module treetaggerwrapper not installed - It is required '
                     'by the italian lemmatizer')

        try:
            self._ttagger = ttagger.TreeTagger(TAGLANG='it',
                                               TAGDIR=treetagger_dir)
        except ttagger.TreeTaggerError as e:
            msg: str = e.args[0]
            if msg.startswith("Can't locate"):
                sys.exit('TreeTagger not installed - It is required by the '
                         'italian lemmatizer')
            elif (msg.startswith('Bad TreeTagger')
                  or msg.startswith('TreeTagger binary')):
                sys.exit('TreeTagger not properly installed - It is required by the '
                         'italian lemmatizer')
            elif msg.startswith('TreeTagger parameter file'):
                sys.exit('TreeTagger italian parameter file not properly '
                         'installed - It is required by the italian lemmatizer')
            else:
                raise

    def lemmatize(self, text: Sequence[str]) -> Generator[Tuple[str, str], None, None]:
        tags = self._ttagger.tag_text(' '.join(text))
        for t in tags:
            sp = t.split('\t')
            if sp[0].lower() in ['sai'] and sp[2] != 'sapere':
                yield (sp[0], 'sapere')
            elif sp[2] == 'essere|stare':
                yield (sp[0], 'stare')
            else:
                yield (sp[0], sp[2])


AVAILABLE_LEMMATIZERS = {
    'en': EnglishLemmatizer,
    'it': ItalianLemmatizer,
}


def get_lemmatizer(lang='en'):
    try:
        return AVAILABLE_LEMMATIZERS[lang]()
    except KeyError:
        pass

    raise ValueError(f'language {lang!r} not available')


def init_argparser():
    """Initialize the command line parser."""
    parser = ArgParse()
    parser.add_argument('datafile', action='store', type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        default='-',
                        help='output file name. If omitted or %(default)r '
                             'stdout is used')
    parser.add_argument('--placeholder', '-p',
                        default=STOPWORD_PLACEHOLDER,
                        help='Placeholder for stopwords. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)r')
    parser.add_argument('--stop-words', '-s',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='stopwords_file',
                        help='stop words file name')
    parser.add_argument('--relevant-term', '-r', nargs='+',
                        metavar=('FILENAME', 'PLACEHOLDER'),
                        dest='relevant_terms_file',
                        action=AppendMultiplePairsAction, unique_first=True,
                        help='relevant terms file name and the placeholder to '
                             'use with those terms. The placeholder must not '
                             'contains any space. The placeholder is optional. '
                             'If it is omitted, each relevant term from this '
                             'file, is replaced with the stopword placeholder, '
                             'followed by the term itself with each space '
                             'changed with the "_" character and then another '
                             'stopword placeholder.', non_standard=True)
    parser.add_argument('--acronyms', '-a',
                        help='TSV files with the approved acronyms')
    parser.add_argument('--target-column', '-t', action='store', type=str,
                        default='abstract', dest='target_column',
                        help='Column in datafile to process. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--output-column', action='store', type=str,
                        default='abstract_lem', dest='output_column',
                        help='name of the column to save. '
                             'If omitted %(default)r is used.')
    parser.add_argument('--input-delimiter', action='store', type=str,
                        default='\t', dest='input_delimiter',
                        help='Delimiter used in datafile. '
                             'Default %(default)r')
    parser.add_argument('--output-delimiter', action='store', type=str,
                        default='\t', dest='output_delimiter',
                        help='Delimiter used in output file. '
                             'Default %(default)r')
    parser.add_argument('--rows', '-R', type=int, dest='input_rows',
                        help="Select maximum number of samples")
    parser.add_argument('--language', '-l', default='en',
                        help='language of text. Must be a ISO 639-1 two-letter '
                             'code. Default: %(default)r')
    parser.add_argument('--logfile', default='slr-kit.log',
                        help='log file name. If omitted %(default)r is used',
                        logfile=True)
    parser.add_argument('--regex', help='Regex .csv for specific substitutions')
    return parser


def load_stopwords(input_file):
    """
    Loads a list of stopwords from a file

    This functions skips all the lines that starts with a '#'.

    :param input_file: file to read
    :type input_file: str
    :return: the loaded words as a set of string
    :rtype: set[str]
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        stop_words_list = f.read().splitlines()

    stop_words_list = {w for w in stop_words_list if w != '' and w[0] != '#'}
    # Creating a list of custom stopwords
    return stop_words_list


def replace_ngram(text, n_grams):
    """
    Replace the given n-grams with a placeholder in the specified text

    The n-grams and their placeholder are taken from the generator n_grams, that
    must be a generator that yields a tuple (placeholder, n-gram). Each n-gram
    must be a tuple of strings.
    The function can also check if the toke immediately after the replaced
    n-gram is equal to the placeholder to remove it. This is useful for acronyms
    because, usually the abbreviation immediately follows the extended acronym.

    :param text: the text to search
    :type text: list[str]
    :param n_grams: generator that yields n-grams and their placeholder
    :type n_grams: Generator[tuple[str, tuple[str]], Any, None]
    :return: the transformed text
    :rtype: list[str]
    """
    text2 = list(text)
    for placeholder, ngram in n_grams:
        end = False
        index = -1
        length = len(ngram)
        while not end:
            try:
                # index + 1 to skip the previous match
                index = text2.index(ngram[0], index + 1)
                if tuple(text2[index:index + length]) == ngram:
                    # found!
                    text2[index:index + length] = [placeholder]
            except ValueError:
                end = True

    return text2


def acronyms_generator(acronyms, prefix_suffix=STOPWORD_PLACEHOLDER):
    """
    Generator that yields acronyms and the relative placeholder for replace_ngram

    The placeholder for each acronym is <prefix_suffix><abbreviation><prefix_suffix>
    The function yields for each acronym: the extended acronym, the extended
    acronym with all the word separated by '-' and the abbreviation of the acronym
    Each acronym is yielded as a tuple of strings.

    :param acronyms: the acronyms to replace in each document. Must have two
        columns 'Acronym' and 'Extended'
    :type acronyms: pd.DataFrame
    :param prefix_suffix: prefix and suffix used to create the placeholder
    :type prefix_suffix: str
    :return: a generator that yields the placeholder and the acronym
    :rtype: Generator[tuple[str, tuple[str]], Any, None]
    """
    for _, row in acronyms.iterrows():
        sub = f'{prefix_suffix}{row["Acronym"]}{prefix_suffix}'
        yield sub, row['Extended']
        alt = ('-'.join(row['Extended']), )
        yield sub, alt
        yield sub, (row['Acronym'], )


def language_specific_regex(text, lang='en'):
    # The first step in every language is to change punctuations to the stop-word
    # placeholder. The stop-word placeholder can be anything, so we have to change
    # the punctuation with something that will survive the special char removal.
    # '---' it's ok because hyphens are preserved in every language.
    if lang == 'en':
        # punctuation
        out_text = re.sub('[,.;:!?()"\']', ' --- ', text)
        # Remove special characters (not the hyphen) and digits
        # also preserve the '__' used by some placeholders
        return re.sub(r'(\d|[^-\w])+|(?<=[^_])_(?=[^_])', ' ', out_text)
    if lang == 'it':
        # punctuation - preserve "'"
        out_text = re.sub('[,.;:!?()"]', ' --- ', text)
        # Remove special characters (not the hyphen) and digits. but preserve
        # accented letters. Also preserve "'" if surrounded by non blank chars
        # and the '__' used by some placeholders
        return re.sub(r'(\d|[^-\wàèéìòù_\']'
                      r'|(?<=\s)\'(?=\S)|(?<=\S)\'(?!\S))+'
                      r'|(?<=[^_])_(?=[^_])',
                      ' ', out_text)


def regex(text, stopword_placeholder=STOPWORD_PLACEHOLDER, lang='en',
          regex_df=None):
    # If a regex DataFrame for the specific project is passed,
    # this function will replace the patterns with the corresponding repl
    # parameter
    if regex_df is not None:
        # Some of the regex are not actually regex. Like <br>
        not_regex = regex_df[~regex_df['regexBoolean']]
        for _, row in not_regex.iterrows():
            text = text.replace(row['pattern'], '__{}__'.format(row['repl']))

        regex_df = regex_df[~regex_df['pattern'].isin(not_regex['pattern'])]
        regex_with_spaces = regex_df[regex_df['pattern'].str.contains(r'\s',
                                                                      regex=False)]

        for _, row in regex_with_spaces.iterrows():
            text = re.sub(row['pattern'], '__{}__'.format(row['repl']), text)

        rows = regex_df[~regex_df['pattern'].isin(regex_with_spaces['pattern'])]
        for _, row in rows.iterrows():
            text = ' '.join([re.sub(row['pattern'],
                                    '__{}__'.format(row['repl']), gram)
                             for gram in text.split()])
    # Change punctuation and remove special characters (not the hyphen) and
    # digits. The definition of special character and punctuation, changes with
    # the language
    text = language_specific_regex(text, lang)
    # now we can search for ' --- ' and place the stop-word placeholder
    # the positive look-ahead and look-behind are to preserve the spaces
    text = re.sub(r'(?<=\s)---(?=\s)', stopword_placeholder, text)
    # remove any run of hyphens not surrounded by non space
    text = re.sub(r'(\s+-+\s+|(?<=\S)-+\s+|\s+-+(?=\S))', ' ', text)

    return text


def relevant_generator(relevant, relevant_prefix):
    for rel_set, placeholder in relevant:
        ph = f'{relevant_prefix}{placeholder}{relevant_prefix}'
        for rel in rel_set:
            if placeholder is not None:
                yield ph, rel
            else:
                yield f'{relevant_prefix}{"_".join(rel)}{relevant_prefix}', rel


def preprocess_item(item, relevant_terms, stopwords, acronyms, language='en',
                    placeholder=STOPWORD_PLACEHOLDER,
                    relevant_prefix=RELEVANT_PREFIX, regex_df=None):
    """
    Preprocess the text of a document.

    It lemmatizes the text. Then it searches for the relevant terms. The
    relevant_terms argument is a list of tuples. Each tuple contains a set of
    relevant terms to search for, and a placeholder. If the placeholder is None,
    each relevant term found is replaced with a string composed with the
    relevant_prefix, then all the words composing the term separated
    with '_' and finally the relevant_prefix. If the placeholder is not None,
    then each term associated with that prefix is replaced with a string
    composed by the relevant_prefix, the placeholder and then the
    relevant_prefix.
    The function then searches for the acronyms, changing the words composing
    them with the corresponding abbreviation.
    It also filters the stop-words changing them with the stopword_placeholder
    string as placeholder.

    :param item: the text to process
    :type item: str
    :param relevant_terms: the relevant words to search. Each n-gram must be a
        tuple of strings
    :type relevant_terms: list[tuple[set[tuple[str]], str or None]]
    :param stopwords: the stop-words to filter
    :type stopwords: set[str]
    :param acronyms: the acronyms to replace in each document. Must have two
        columns 'Acronym' and 'Extended'
    :type acronyms: pd.DataFrame
    :param language: code of the language to be used to lemmatize text
    :type language: str
    :param placeholder: placeholder for the stop-words
    :type placeholder: str
    :param relevant_prefix: prefix string used when replacing the relevant terms
    :type relevant_prefix: str
    :param regex_df: dataframe with the regex to apply
    :type regex_df: pd.DataFrame
    :return: the processed text
    :rtype: list[str]
    """
    lem = get_lemmatizer(language)
    # Convert to lowercase
    text = item.lower()
    # apply some regex to clean the text
    text = regex(text, placeholder, language, regex_df=regex_df)
    text = text.split(' ')

    # replace acronyms
    text = replace_ngram(text, acronyms_generator(acronyms, relevant_prefix))

    text2 = [lem_word for _, lem_word in lem.lemmatize(text)]

    # mark relevant terms
    text2 = replace_ngram(text2,
                          relevant_generator(relevant_terms, relevant_prefix))

    for i, word in enumerate(text2):
        if word in stopwords:
            text2[i] = placeholder

    text2 = ' '.join(text2)
    return text2


def process_corpus(dataset, relevant_terms, stopwords, acronyms, language='en',
                   placeholder=STOPWORD_PLACEHOLDER,
                   relevant_prefix=RELEVANT_PREFIX, regex_df=None):
    """
    Process a corpus of documents.

    Each documents is processed using the preprocess_item function

    :param dataset: the corpus of documents
    :type dataset: pd.Sequence
    :param relevant_terms: the related terms to search in each document
    :type relevant_terms: list[tuple[set[tuple[str]], str]]
    :param stopwords: the stop-words to filter in each document
    :type stopwords: set[str]
    :param acronyms: the acronyms to replace in each document. Must have two
        columns 'Acronym' and 'Extended'
    :type acronyms: pd.Dataframe
    :param language: code of the language to be used to lemmatize text
    :type language: str
    :param placeholder: placeholder for the stop-words
    :type placeholder: str
    :param relevant_prefix: prefix used to replace the relevant terms
    :type relevant_prefix: str
    :param regex_df: dataframe with the regex to apply
    :type regex_df: pd.DataFrame or None
    :return: the corpus processed
    :rtype: list[str]
    """
    with Pool(processes=PHYSICAL_CPUS) as pool:
        corpus = pool.starmap(preprocess_item, zip(dataset,
                                                   repeat(relevant_terms),
                                                   repeat(stopwords),
                                                   repeat(acronyms),
                                                   repeat(language),
                                                   repeat(placeholder),
                                                   repeat(relevant_prefix),
                                                   repeat(regex_df)))
    return corpus


def load_relevant_terms(input_file):
    """
    Loads a list of relevant terms from a file

    This functions skips all the lines that starts with a '#'.
    Each term is split in a tuple of strings

    :param input_file: file to read
    :type input_file: str
    :return: the loaded terms as a set of tuples of strings
    :rtype: set[tuple[str]]
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        rel_words_list = f.read().splitlines()

    rel_words_list = {tuple(w.split(' '))
                      for w in rel_words_list
                      if w != '' and w[0] != '#'}

    return rel_words_list


def preprocess(args):
    if args.language not in AVAILABLE_LEMMATIZERS:
        print(f'Language {args.language!r} is not available', file=sys.stderr)
        sys.exit(1)
    else:
        # try to instantiate the lemmatizer to detect import/install errors
        # before starting the multiprocess machinery
        get_lemmatizer(args.language)

    target_column = args.target_column
    debug_logger = setup_logger('debug_logger', args.logfile,
                                level=logging.DEBUG)
    name = 'preprocess'
    log_start(args, debug_logger, name)

    # load the dataset
    dataset = pd.read_csv(args.datafile, delimiter=args.input_delimiter,
                          encoding='utf-8', nrows=args.input_rows)
    dataset.fillna('', inplace=True)
    assert_column(args.datafile, dataset, target_column)
    debug_logger.debug('Dataset loaded {} items'.format(len(dataset[target_column])))

    # csvFileName da CLI
    if args.regex is not None:
        regex_df = pd.read_csv(args.regex, sep=',', quotechar='"')
    else:
        regex_df = None

    placeholder = args.placeholder
    relevant_prefix = placeholder

    stopwords = set()
    if args.stopwords_file is not None:
        for sfile in args.stopwords_file:
            stopwords |= load_stopwords(sfile)

        debug_logger.debug('Stop-words loaded and updated')

    if args.acronyms is not None:
        conv = {
            'Acronym': lambda s: s.lower(),
            'Extended': lambda s: tuple(s.lower().split(' ')),
        }
        acronyms = pd.read_csv(args.acronyms, delimiter='\t', encoding='utf-8',
                               converters=conv)
        assert_column(args.acronyms, acronyms, ['Acronym', 'Extended'])
        debug_logger.debug('Acronyms loaded and updated')
    else:
        acronyms = pd.DataFrame(columns=['Acronym', 'Extended'])

    rel_terms = []
    if args.relevant_terms_file is not None:
        for rfile, placeholder in args.relevant_terms_file:
            if placeholder is not None and ' ' in placeholder:
                sys.exit('A relevant term placeholder can not contain spaces')

            rel_terms.append((load_relevant_terms(rfile), placeholder))

        debug_logger.debug('Relevant words loaded and updated')

    start = timer()
    corpus = process_corpus(dataset[target_column], rel_terms, stopwords,
                            acronyms, language=args.language,
                            placeholder=placeholder,
                            relevant_prefix=relevant_prefix, regex_df=regex_df)
    stop = timer()
    elapsed_time = stop - start
    debug_logger.debug('Corpus processed')
    dataset[args.output_column] = corpus

    # write to output, either a file or stdout (default)
    if args.output == '-':
        output_file = sys.stdout
    else:
        output_file = open(args.output, 'w', encoding='utf-8')

    dataset.to_csv(output_file, index=None, header=True,
                   sep=args.output_delimiter)
    print('Elapsed time:', elapsed_time, file=sys.stderr)
    debug_logger.info(f'elapsed time: {elapsed_time}')
    log_end(debug_logger, name)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()
