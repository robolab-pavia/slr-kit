import pathlib

import pandas

import sys
import csv
import logging
import argparse
from utils import setup_logger

BARRIER_PLACEHOLDER = 'XXX'
RELEVANT_PREFIX = BARRIER_PLACEHOLDER


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action='store', type=str,
                        help='Input CSV data file')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='Output file name')
    parser.add_argument('--n-grams', '-n', metavar='N', dest='n_grams',
                        default=4, help='Maximum size of n-grams number')
    parser.add_argument('--min-frequency', '-m', metavar='N',
                        dest='min_frequency', default=5,
                        help='Minimum frequency of the n-grams')
    parser.add_argument('--column', '-c', default='abstract_lem',
                        help='Column in datafile to process. '
                             'If omitted %(default)r is used.')
    return parser


def get_n_grams(corpus, n_terms=1, min_frequency=5, barrier=None,
                relevant_prefix=None):
    """
    Extracts n-grams from the corpus.

    The output is a dict of n-grams, where each dict item key
    is the n-gram and the value its the frequency, sorted by frequency.

    :param corpus: the text to extract terms from
    :type corpus: list[str]
    :param n_terms: maximum length of an n-gram in number of words
    :type n_terms: int
    :param min_frequency: min. number of occurrences. n-grams with less than
        this frequency are discarded
    :type min_frequency: int
    :param barrier: placeholder for the barrier word. No n-gram with this
        placeholder is returned
    :type barrier: str or None
    :param relevant_prefix: prefix used to mark the relevant terms. No n-gram
        containing a word with this prefix is returned
    :type relevant_prefix: str or None
    :return: the n-grams as a dict with the n-gram itself as the key and its
        frequency as the value
    :rtype: dict[str, int]
    """
    terms = {}
    if relevant_prefix is None:
        # no prefix, use something that startswith can't find
        relevant_prefix = ' '

    for doc in corpus:
        doc_list = doc.split(' ')
        for i in range(len(doc_list) - n_terms + 1):
            words = doc_list[i:i+n_terms]
            # skip the terms that contain a barrier or a relevant term
            if any(word == barrier or word.startswith(relevant_prefix)
                   for word in words):
                continue

            term = ' '.join(words)
            if term in terms:
                terms[term] += 1
            else:
                terms[term] = 1

    limited_terms = {k: v for k, v in terms.items() if v >= min_frequency}
    sorted_dict = {k: v for k, v in sorted(limited_terms.items(),
                                           key=lambda item: item[1],
                                           reverse=True)}
    return sorted_dict


def convert_int_parameter(args, arg_name, default=None):
    """Try to convert an integer number from the command line argument.

    Checks the validity of the value.
    Assigns an optional default value if no option is provided.
    Exits in case of error during the conversion.
    """
    arg_string = args.__dict__[arg_name]
    if arg_string is not None:
        try:
            value = int(arg_string)
        except ValueError:
            msg = 'Invalid value for parameter "{}": "{}"'
            print(msg.format(arg_name, arg_string))
            sys.exit(1)
    else:
        value = default
    return value


def main():
    parser = init_argparser()
    args = parser.parse_args()
    target_column = args.column

    # set the value of n_grams, possibly from the command line
    n_grams = convert_int_parameter(args, 'n_grams', default=4)
    min_frequency = convert_int_parameter(args, 'min_frequency', default=5)

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)
    # TODO: write log string with values of the parameters used in the execution

    # load the dataset
    dataset = pandas.read_csv(args.datafile, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    if target_column not in dataset:
        msg = 'File "{}" must contain a column labelled as "{}".'
        print(msg.format(args.datafile, target_column))
        sys.exit(1)

    msg = 'Dataset loaded {} items'
    debug_logger.debug(msg.format(len(dataset[target_column])))
    corpus = dataset[target_column].to_list()

    list_of_grams = []
    for n in range(1, n_grams + 1):
        top_terms = get_n_grams(corpus, n_terms=n, min_frequency=min_frequency,
                                barrier=BARRIER_PLACEHOLDER,
                                relevant_prefix=RELEVANT_PREFIX)
        list_of_grams.append(top_terms)

    if args.output is not None:
        output_file = open(args.output, 'w', encoding='utf-8')
        path = pathlib.Path(args.output)
        out = path.stem
        name = '_'.join([out, 'fawoc_data.tsv'])
        fawoc_file = open(path.parent / name, 'w')
    else:
        output_file = sys.stdout
        fawoc_file = open('fawoc_data.tsv', 'w')

    fawoc_data = []
    writer = csv.writer(output_file, delimiter='\t', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'term', 'label'])
    index = 0
    for terms in list_of_grams:
        for key in terms:
            writer.writerow([index, key, ''])
            fawoc_data.append({
                'id': index,
                'term': key,
                'count': int(terms[key]),
            })
            index += 1

    fawoc_data_writer = csv.DictWriter(fawoc_file, delimiter='\t',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                       fieldnames=fawoc_data[0].keys())
    fawoc_data_writer.writeheader()
    fawoc_data_writer.writerows(fawoc_data)
    fawoc_file.close()
    if output_file is not sys.stdout:
        output_file.close()


if __name__ == '__main__':
    main()
