import logging
import os
import string
import json
import sys

STOPWORD_PLACEHOLDER = '@'
RELEVANT_PREFIX = STOPWORD_PLACEHOLDER


def load_df(filename, required_columns=None):
    """
    Loads pandas dataframe from TAB-separated CSV.

    Optionally, it checks the presence of some required columns
    and exits in case of missing columns.

    :param filename: name of the input file
    :type filename: str
    :param required_columns: list of names of required columns
    :type required_columns: list[str] or None
    :return: the loaded dataframe
    :rtype: pd.DataFrame
    """
    import pandas as pd
    input_df = pd.read_csv(filename, delimiter='\t', encoding='utf-8')
    input_df.fillna('', inplace=True)
    for col_name in required_columns:
        assert_column(filename, input_df, col_name)

    return input_df


def load_dtj(infile):
    """
    Load the document-terms JSON file format.

    :param infile: name of the input file
    :type infile: str
    :return: the list of parsed rows
    :rtype: list[dict]
    """
    dtj = []
    with open(infile, encoding='utf-8') as input_file:
        for line in input_file:
            dtj.append(json.loads(line))

    return dtj


def assert_column(infile, dataframe, column_names):
    """
    Check a datafile for the presence of columns.

    Many scripts require that some specific column are
    present in the datafile under processing. This
    function can be used to make sure that the script
    does not hang due to the absence of the necessary
    columns.

    :param infile: name of the input file
    :type infile: str
    :param dataframe: dataframe containing the required column
    :type dataframe: pd.DataFrame or dict
    :param column_names: name of the required column
    :type column_names: str or list[str]
    """
    if not isinstance(column_names, list):
        column_names = [column_names]

    for c in column_names:
        if c not in dataframe:
            msg = 'File {!r} must contain the {!r} column.'
            sys.exit(msg.format(infile, c))


def setup_logger(name, log_file,
                 formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """
    Function to setup a generic loggers.

    :param name: name of the logger
    :type name: str
    :param log_file: file of the log
    :type log_file: str
    :param formatter: formatter to be used by the logger
    :type formatter: logging.Formatter
    :param level: level to display
    :type level: int
    :return: the logger
    :rtype: logging.Logger
    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def substring_index(haystack, needle, delim=string.whitespace):
    """
    Generator function that give the slices of haystack where needle is found

    needle is considered found if it is between two word boundaries.
    Example:
    haystack = 'the rate monotonic scheduling algorithm to meet the deadlines'
    needle = 'to'
    the generator yields only the 'to' between 'algorithm' and 'meet' not the
    'to' inside 'monoTOnic'.
    delim specify which characters are to be considered boundaries of a word.
    Default uses all whitespace characters
    The values yielded are in the form of a tuple (begin, end) with
    haystack[begin:end] == needle
    If needle is not found the generator raises immediately StopIteration
    :param haystack: the string to examine
    :type haystack: str
    :param needle: the string to search
    :type needle: str
    :param delim: character used as word boundaries
    :type delim: str or list[str]
    :return: a generator that yields the slice indexes where needle is found
    """
    if needle == '':
        return

    if haystack == needle:
        yield (0, len(haystack))
        return

    if isinstance(delim, list):
        delim = ''.join(delim)

    start = 0
    idx = haystack.find(needle, start)
    while idx >= 0:
        start = idx + len(needle)
        if idx == 0:
            if haystack[len(needle)] in delim:
                yield (0, len(needle))
        elif haystack[idx - 1] in delim:
            if idx + len(needle) == len(haystack):
                yield (idx, len(haystack))
                return
            elif haystack[idx + len(needle)] in delim:
                yield (idx, idx + len(needle))

        idx = haystack.find(needle, start)


def substring_check(haystack, needle, delim=string.whitespace):
    """
    Tells if haystack contains at least one instance of needle between two word boundaries

    delim specify which characters are to be considered boundaries of a word.
    Default uses all whitespace characters
    :param haystack: the string to examine
    :type haystack: str
    :param needle: the string to search
    :type needle: str
    :param delim: character used as word boundaries
    :type delim: str or list[str]
    :return: True if needle is found
    :rtype: bool
    """
    for _ in substring_index(haystack, needle, delim=delim):
        return True
    else:
        return False


def log_start(args, debug_logger, name):
    debug_logger.info(f'=== {name} started ===')
    debug_logger.info(f'cwd {os.getcwd()}')
    msg = 'arguments:'
    for k, v in vars(args).items():
        msg = f'{msg} {k!r}: {v!r}'
    debug_logger.info(msg)


def log_end(debug_logger, name):
    debug_logger.info(f'=== {name} ended ===')
