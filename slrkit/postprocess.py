import re
import sys

import tomlkit

# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

import logging
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from psutil import cpu_count

import pandas as pd

from slrkit_utils.argument_parser import ArgParse
from utils import substring_index, STOPWORD_PLACEHOLDER, assert_column


PHYSICAL_CPUS = cpu_count(logical=False)


# TODO: old stuff to clean up
def to_ignore(_):
    return ['lda*.json', 'lda_info*.txt']


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    epilog = "This script outputs the topics in " \
             "<outdir>/<project>_postprocess.csv"
    parser = ArgParse(description='Filters the documents to keep only '
                                  'relevant/keyword terms',
                      epilog=epilog)
    parser.add_argument('preproc_file', action='store', type=Path,
                        help='path to the the preprocess file with the text '
                             'to elaborate.', input=True)
    parser.add_argument('terms_file', action='store', type=Path,
                        help='path to the file with the classified terms.',
                        input=True)
    parser.add_argument('outdir', action='store', type=Path, nargs='?',
                        default=Path.cwd(), input=True,
                        help='path to the directory where '
                             'to save the results.')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        default='-',
                        help='Output file name. Default or %(default)r '
                             'stdout is used',
                        suggest_suffix='_postproc.csv', output=True)
    parser.add_argument('--text-column', '-t', action='store', type=str,
                        default='abstract_lem', dest='target_column',
                        help='Column in preproc_file to process. '
                             'Default %(default)r.')
    parser.add_argument('--title-column', action='store', type=str,
                        default='title', dest='title_column',
                        help='Column in preproc_file to use as document '
                             ' title. Default %(default)r.')
    parser.add_argument('--filtered-column', action='store', type=str,
                        default='abstract_filtered', dest='filtered_column',
                        help='Column to save filtered data in postproc_file '
                             ' title. Default %(default)r.')
    parser.add_argument('--no-relevant', action='store_true',
                        help='if set, use only the term labelled as keyword')
    parser.add_argument('--placeholder', '-p',
                        default=STOPWORD_PLACEHOLDER,
                        help='Placeholder for barrier word. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)s')
    parser.add_argument('--delimiter', action='store', type=str,
                        default='\t',
                        help='Delimiter used in preproc and postproc files. '
                             'Default %(default)r')
    parser.add_argument('--config', '-c', action='store', type=Path,
                        help='Path to a toml config file like the one used by '
                             'the slrkit postprocess command. It overrides '
                             'all the CLI arguments.', cli_only=True)
    return parser


def load_term_data(terms_file):
    words_dataset = pd.read_csv(terms_file, delimiter='\t',
                                encoding='utf-8')
    try:
        terms = words_dataset['term'].to_list()
    except KeyError:
        terms = words_dataset['keyword'].to_list()
    term_labels = words_dataset['label'].to_list()
    return term_labels, terms


def load_ngrams(terms_file, labels=('keyword', 'relevant')):
    term_labels, terms = load_term_data(terms_file)
    zipped = zip(terms, term_labels)
    good = [x[0] for x in zipped if x[1] in labels]
    ngrams = {1: set()}
    for x in good:
        n = x.count(' ') + 1
        try:
            ngrams[n].add(x)
        except KeyError:
            ngrams[n] = {x}

    return ngrams


def check_ngram(doc, idx):
    for dd in doc:
        r = range(dd[0][0], dd[0][1])
        yield idx[0] in r or idx[1] in r


def filter_doc(d: str, ngram_len, terms, placeholder, relevant_prefix):
    additional = []
    end = False
    idx = 0
    d = re.sub(rf'\B{placeholder}\B', ' ', d)
    while not end:
        i = d.find(relevant_prefix, idx)
        if i < 0:
            end = True
        else:
            stop = d.find(relevant_prefix, i+1)
            if stop > 0:
                additional.append(((i+1, stop), d[i+1:stop]))
                idx = stop + 1
            else:
                end = True

    doc = []
    flag = False
    for n in ngram_len:
        for t in terms[n]:
            for idx in substring_index(d, t):
                if flag and any(check_ngram(doc, idx)):
                    continue

                doc.append((idx, t.replace(' ', '_')))

        flag = True

    doc.extend(additional)
    doc.sort(key=lambda dd: dd[0])
    return [t[1] for t in doc]


def load_terms(terms_file, labels=('keyword', 'relevant')):
    term_labels, terms = load_term_data(terms_file)
    zipped = zip(terms, term_labels)
    good = [x for x in zipped if x[1] in labels]
    good_set = set()
    for x in good:
        good_set.add(x[0])

    return good_set


def linear_filtering(documents, ngram_len,
                     terms, placeholder, relevant_prefix):
    logger = logging.getLogger('debug_logger')
    docs = []
    for i, d in enumerate(documents):
        result = filter_doc(d, ngram_len, terms, placeholder, relevant_prefix)
        docs.append(result)
        logger.debug(f'Completed document: {i}/{len(documents)}')
    return docs


def parallel_filtering(documents, ngram_len,
                       terms, placeholder, relevant_prefix):
    with Pool(processes=PHYSICAL_CPUS) as pool:
        docs = pool.starmap(filter_doc,
                            zip(documents,
                                repeat(ngram_len),
                                repeat(terms),
                                repeat(placeholder),
                                repeat(relevant_prefix)))
    return docs


def filter_docs(terms_file, preproc_file,
                target_col='abstract_lem',
                title_col='title',
                filtered_col='filtered_abstract',
                delimiter='\t',
                labels=('keyword', 'relevant'),
                placeholder=STOPWORD_PLACEHOLDER,
                relevant_prefix=STOPWORD_PLACEHOLDER):
    logger = logging.getLogger('debug_logger')
    terms = load_ngrams(terms_file, labels)
    ngram_len = sorted(terms, reverse=True)
    docs = load_documents(preproc_file, target_col, delimiter)
    msg = f"Filtering data from '{target_col}' in {preproc_file}"
    src_docs = docs[target_col].to_list()
    logger.debug(msg)
    # filtered_docs = linear_filtering(src_docs, ngram_len, terms, placeholder, relevant_prefix)
    filtered_docs = parallel_filtering(src_docs, ngram_len, terms, placeholder, relevant_prefix)
    joined = []
    for elem in filtered_docs:
        joined.append(' '.join(elem))
    docs[filtered_col] = joined
    return docs


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

    rel_words_list = {w.replace(' ', '_')
                      for w in rel_words_list
                      if w != '' and w[0] != '#'}
    return rel_words_list


def load_documents(preproc_file, target_col, delimiter):
    try:
        dataset = pd.read_csv(preproc_file,
                              delimiter=delimiter,
                              encoding='utf-8')
    except FileNotFoundError as err:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(err.filename))

    assert_column(str(preproc_file), dataset, [target_col])
    dataset.fillna('', inplace=True)
    return dataset


def postprocess(args):
    logger = logging.getLogger('debug_logger')
    terms_file = args.terms_file
    preproc_file = args.preproc_file
    output_dir = args.outdir
    output_dir.mkdir(exist_ok=True)

    placeholder = args.placeholder
    relevant_prefix = placeholder

    if args.no_relevant:
        labels = ('keyword',)
    else:
        labels = ('keyword', 'relevant')

    df = filter_docs(terms_file, preproc_file,
                     target_col=args.target_column,
                     title_col=args.title_column,
                     filtered_col=args.filtered_column,
                     delimiter=args.delimiter,
                     labels=labels,
                     placeholder=placeholder,
                     relevant_prefix=relevant_prefix)
    if args.output == '-':
        output_file = sys.stdout
    else:
        if args.outdir is not None:
            output_file_name = args.outdir / args.output
        else:
            output_file_name = args.output
        output_file = open(output_file_name, 'w', encoding='utf-8')
        msg = (
            f"Saving filtered data in column "
            f"'{args.filtered_column}' of file {args.output}"
        )
        print(msg)
        logger.debug(msg)
    df.to_csv(output_file, sep=args.delimiter,
              encoding='utf-8', index_label='id')


def main():
    parser = init_argparser()
    args = parser.parse_args()
    if args.config is not None:
        try:
            with open(args.config) as file:
                config = tomlkit.loads(file.read())
        except FileNotFoundError:
            msg = 'Error: config file {} not found'
            sys.exit(msg.format(args.config))

        from slrkit import prepare_script_arguments
        args, _, _ = prepare_script_arguments(config, args.config.parent,
                                              args.config.name,
                                              parser.slrkit_arguments)
        # handle the outdir parameter
        param = Path(config.get('outdir', Path.cwd()))
        setattr(args, 'outdir', param.resolve())
    postprocess(args)


if __name__ == '__main__':
    main()
