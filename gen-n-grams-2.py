import pathlib

import pandas

import sys
import csv
import logging
import argparse
from utils import setup_logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="Input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='Output file name')
    parser.add_argument('--n-grams', '-n', metavar='N', dest='n_grams', default=4,
                        help='Maximum size of n-grams number')
    parser.add_argument('--min-frequency', '-m', metavar='N', dest='min_frequency',
                        default=5,
                        help='Minimum frequency of the n-grams')
    return parser


def get_n_grams(corpus, n_terms=1, min_frequency=5, barrier=None):
    """Extracts n-grams from the corpus.

    The output is a dict of n-grams, wher each dict item key
    is the n-gram and the value its the frequency, sorted by frequency.
    """
    terms = {}
    for doc in corpus:
        doc_list = doc.split(' ')
        for i in range(len(doc_list) - n_terms + 1):
            words = doc_list[i:i+n_terms]
            # skip the terms that contain a barrier
            if barrier in words:
                continue
            term = ' '.join(words)
            if term in terms:
                terms[term] += 1
            else:
                terms[term] = 1
    limited_terms = {k: v for k, v in terms.items() if v >= min_frequency}
    sorted_dict = {k: v for k, v in sorted(limited_terms.items(), key=lambda item: item[1], reverse=True)}
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
        except:
            print('Invalid value for parameter "{}": "{}"'.format(arg_name, arg_string))
            sys.exit(1)
    else:
        value = default
    return value


def main():
    target_column = 'abstract_lem'
    parser = init_argparser()
    args = parser.parse_args()

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
        print('File "{}" must contain a column labelled as "{}".'.format(args.datafile, target_column))
        sys.exit(1)
    debug_logger.debug("Dataset loaded {} items".format(len(dataset[target_column])))
    #logging.debug(dataset.head())

    corpus = dataset[target_column].to_list()

    # logging.debug(corpus[2])

    list_of_grams = []
    for n in range(1, n_grams + 1):
        top_terms = get_n_grams(corpus, n_terms=n, min_frequency=min_frequency, barrier='XXX')
        #print(len(top_terms))
        list_of_grams.append(top_terms)


    if args.output is not None:
        output_file = open(args.output, 'w', encoding='utf-8')
        out = pathlib.Path(args.output).stem
        name = '_'.join([out, 'fawoc_data.tsv'])
        fawoc_file = open(name, 'w')
    else:
        output_file = sys.stdout
        fawoc_file = open('fawoc_data.tsv', 'w')

    fawoc_data = []
    writer = csv.writer(output_file, delimiter='\t', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'keyword', 'label'])
    index = 0
    for terms in list_of_grams:
        for key in terms:
            writer.writerow([index, key, ''])
            fawoc_data.append({
                'id': index,
                'count': int(terms[key]),
            })
            index += 1


    #for i, item in enumerate(all_terms):
    #    writer.writerow([i, item[0], ''])
    #    fawoc_data.append({
    #        'id': i,
    #        'count': int(item[1]),
    #    })
    fawoc_data_writer = csv.DictWriter(fawoc_file, delimiter='\t',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                       fieldnames=fawoc_data[0].keys())
    fawoc_data_writer.writeheader()
    fawoc_data_writer.writerows(fawoc_data)
    fawoc_file.close()
    if output_file is not sys.stdout:
        output_file.close()


    return



    # write to output, either a file or stdout (default)
    # TODO: use pandas to_csv instead of explicit csv row output
    output_file = open(args.output, 'w',
                       encoding='utf-8') if args.output is not None else sys.stdout
    writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['keyword', 'count', 'label'])
    for terms in list_of_grams:
        for key in terms:
            writer.writerow([key, terms[key], ''])
    if output_file is not sys.stdout:
        output_file.close()


if __name__ == "__main__":
    main()
