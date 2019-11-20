import logging


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
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


def substring_index(haystack, needle):
    """
    Generator function that give the slices of haystack where needle is found

    needle is considered found if it is between two word boundaries.
    Example:
    haystack = 'the rate monotonic scheduling algorithm to meet the deadlines'
    needle = 'to'
    the generator yields only the 'to' between 'algorithm' and 'meet' not the
    'to' inside 'monoTOnic'.
    The values yielded are in the form of a tuple (begin, end) with
    haystack[begin:end] == needle
    If needle is not found the generator raises immediately StopIteration
    :param haystack: the string to examine
    :type haystack: str
    :param needle: the string to search
    :type needle: str
    :return: a generator that yields the slice indexes where needle is found
    """
    if needle == '':
        return

    if haystack == needle:
        yield (0, len(haystack))

    start = 0
    idx = haystack.find(needle, start)
    while idx >= 0:
        start = idx + len(needle)
        if idx == 0:
            if haystack[len(needle)] == ' ':
                yield (0, len(needle))
        elif haystack[idx - 1] == ' ':
            if idx + len(needle) == len(haystack):
                yield (idx, len(haystack))
            elif haystack[idx + len(needle)] == ' ':
                yield (idx, idx + len(needle))

        idx = haystack.find(needle, start)
