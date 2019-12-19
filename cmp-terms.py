import argparse
import json
import logging
import os
import sys
from terms import Label, TermList


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


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('oldfile', action="store", type=str,
                        help="older CSV data file")
    parser.add_argument('newfile', action="store", type=str,
                        help="newer CSV data file")
    return parser


def main():
    """
    Main function
    """
    parser = init_argparser()
    args = parser.parse_args()

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    # loads the two datasets
    oldterms = TermList()
    _, _ = oldterms.from_csv(args.oldfile)
    newterms = TermList()
    _, _ = newterms.from_csv(args.newfile)

    # compare the older dataset with the newer one
    max_label_len = max([len(item.term) for item in oldterms.items])
    for i, olditem in enumerate(oldterms.items):
        newitem = newterms.items[i]
        # check whether the two datasets are coherent
        if olditem.term != newitem.term:
            print('Files do not match at line {}: "{}" <> "{}"'.format(i, olditem.term, newitem.term))
            return
        # compares the items at the same line
        if olditem.label != newitem.label:
            padding = '.' * (max_label_len - len(olditem.term))
            print('{:02} {} {} "{}" -> "{}"'.format(i, olditem.term, padding, olditem.label.label_name, newitem.label.label_name))


if __name__ == "__main__":
    main()
