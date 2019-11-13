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
