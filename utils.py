import logging
import string
import sys


def assert_column(infile, dataframe, column_name):
    """
    Check a datafile for the presence of a column.

    Many scripts require that a specific column is present
    in the datafile under processing. This function can be
    to make sure that the script does not hang due to the
    absence of the necessary column.

    :param infile: name of the input file
    :type name: str
    :param dataframe: dataframe containing the required column
    :type name: pandas dataframe or any dict-accessible variable
    :param column_name: name of the required column
    :type name: str
    """
    if column_name not in dataframe:
        print('File "{}" must contain the "{}" column.'.format(infile, column_name))
        sys.exit(1)


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
