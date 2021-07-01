r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""
import sys
# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

import argparse
import json
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from psutil import cpu_count

from slrkit_utils.argument_parser import AppendMultipleFilesAction, ArgParse
from utils import (substring_index, STOPWORD_PLACEHOLDER, RELEVANT_PREFIX,
                   assert_column)

PHYSICAL_CPUS = cpu_count(logical=False)


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    epilog = "This script outputs the topics in " \
             "<outdir>/lda_terms-topics_<date>_<time>.json and the topics" \
             "assigned to each document in" \
             "<outdir>/lda_docs-topics_<date>_<time>.json"
    parser = ArgParse(description='Performs the LDA on a dataset', epilog=epilog)
    parser.add_argument('preproc_file', action='store', type=Path,
                        help='path to the the preprocess file with the text to '
                             'elaborate.')
    parser.add_argument('terms_file', action='store', type=Path,
                        help='path to the file with the classified terms.')
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
    parser.add_argument('--additional-terms', '-T',
                        action=AppendMultipleFilesAction, nargs='+',
                        metavar='FILENAME', dest='additional_file',
                        help='Additional keywords files')
    parser.add_argument('--acronyms', '-a',
                        help='TSV files with the approved acronyms')
    parser.add_argument('--topics', action='store', type=int,
                        default=20, help='Number of topics. If omitted '
                                         '%(default)s is used')
    parser.add_argument('--alpha', action='store', type=str,
                        default='auto', help='alpha parameter of LDA. If '
                                             'omitted %(default)s is used')
    parser.add_argument('--beta', action='store', type=str,
                        default='auto', help='beta parameter of LDA. If omitted'
                                             ' %(default)s is used')
    parser.add_argument('--no_below', action='store', type=int,
                        default=20, help='Keep tokens which are contained in at'
                                         ' least this number of documents. If '
                                         'omitted %(default)s is used')
    parser.add_argument('--no_above', action='store', type=float,
                        default=0.5, help='Keep tokens which are contained in '
                                          'no more than this fraction of '
                                          'documents (fraction of total corpus '
                                          'size, not an absolute number). If '
                                          'omitted %(default)s is used')
    parser.add_argument('--seed', type=int, help='Seed to be used in training')
    parser.add_argument('--no-ngrams', action='store_true',
                        help='if set do not use the ngrams')
    parser.add_argument('--model', action='store_true',
                        help='if set, the lda model is saved to directory '
                             '<outdir>/lda_model. The model is saved '
                             'with name "model.')
    parser.add_argument('--no-relevant', action='store_true',
                        help='if set, use only the term labelled as keyword')
    parser.add_argument('--load-model', action='store',
                        help='Path to a directory where a previously trained '
                             'model is saved. Inside this directory the model '
                             'named "model" is searched. the loaded model is '
                             'used with the dataset file to generate the topics'
                             ' and the topic document association')
    parser.add_argument('--placeholder', '-p',
                        default=STOPWORD_PLACEHOLDER,
                        help='Placeholder for barrier word. Also used as a '
                             'prefix for the relevant words. '
                             'Default: %(default)s')
    parser.add_argument('--delimiter', action='store', type=str,
                        default='\t', help='Delimiter used in preproc_file. '
                                           'Default %(default)r')
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


def filter_doc(d, ngram_len, terms):
    doc = []
    flag = False
    for n in ngram_len:
        for t in terms[n]:
            for idx in substring_index(d, t):
                if flag and any(check_ngram(doc, idx)):
                    continue

                doc.append((idx, t.replace(' ', '_')))

        flag = True
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


def generate_filtered_docs_ngrams(terms_file, preproc_file,
                                  target_col='abstract_lem', title_col='title',
                                  delimiter='\t', labels=('keyword', 'relevant'),
                                  additional=None, acronyms=None,
                                  placeholder=STOPWORD_PLACEHOLDER,
                                  relevant_prefix=STOPWORD_PLACEHOLDER):
    terms = load_ngrams(terms_file, labels)
    keywords = []
    if additional is not None:
        keywords.extend(additional)

    if acronyms is not None:
        keywords.extend(acronyms)

    for kw in keywords:
        n = kw.count(' ') + 1
        try:
            terms[n].add(kw)
        except KeyError:
            # no terms with n words: add this category to include the
            # additional keyword
            terms[n] = {kw}

    ngram_len = sorted(terms, reverse=True)
    documents, titles = load_documents(preproc_file, target_col, title_col,
                                       delimiter, placeholder, relevant_prefix)
    with Pool(processes=PHYSICAL_CPUS) as pool:
        docs = pool.starmap(filter_doc, zip(documents, repeat(ngram_len),
                                            repeat(terms)))

    return docs, titles


def generate_filtered_docs(terms_file, preproc_file, target_col='abstract_lem',
                           title_col='title', delimiter='\t',
                           labels=('keyword', 'relevant'), additional=None,
                           acronyms=None, placeholder=STOPWORD_PLACEHOLDER,
                           relevant_prefix=STOPWORD_PLACEHOLDER):
    if additional is None:
        additional = set()

    if acronyms is not None:
        additional |= {acro for acro in acronyms}

    terms = load_terms(terms_file, labels) | additional
    documents, titles = load_documents(preproc_file, target_col, title_col,
                                       delimiter, placeholder, relevant_prefix)
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in terms]
        good_docs.append(gd)

    return good_docs, titles


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


def prepare_documents(preproc_file, terms_file, ngrams, labels,
                      target_col='abstract_lem', title_col='title',
                      delimiter='\t', additional_keyword=None,
                      acronyms=None, placeholder=STOPWORD_PLACEHOLDER,
                      relevant_prefix=STOPWORD_PLACEHOLDER):
    """
    Elaborates the documents preparing the bag of word representation

    :param preproc_file: path to the csv file with the lemmatized abstracts
    :type preproc_file: str
    :param terms_file: path to the csv file with the classified terms
    :type terms_file: str
    :param target_col: name of the column in preproc_file with the document text
    :type target_col: str
    :param title_col: name of the column used as document title
    :type title_col: str
    :param delimiter: delimiter used in preproc_file
    :type delimiter: str
    :param ngrams: if True use all the ngrams
    :type ngrams: bool
    :param labels: use only the terms classified with the labels specified here
    :type labels: tuple[str]
    :param additional_keyword: additional keyword loaded from file
    :type additional_keyword: set or None
    :param acronyms: list of approved acronyms to be considered as keyword
    :type acronyms: list or None
    :param placeholder: placeholder for stop-words
    :type placeholder: str
    :param relevant_prefix: prefix used to mark relevant terms
    :type relevant_prefix: str
    :return: the documents as bag of words and the document titles
    :rtype: tuple[list[list[str]], list[str]]
    """
    if additional_keyword is None:
        additional_keyword = set()
    if ngrams:
        ret = generate_filtered_docs_ngrams(terms_file, preproc_file,
                                            target_col, title_col, delimiter,
                                            labels,
                                            additional=additional_keyword,
                                            acronyms=acronyms,
                                            placeholder=placeholder,
                                            relevant_prefix=relevant_prefix)
    else:
        ret = generate_filtered_docs(terms_file, preproc_file, target_col,
                                     title_col, delimiter, labels,
                                     additional=additional_keyword,
                                     acronyms=acronyms, placeholder=placeholder,
                                     relevant_prefix=relevant_prefix)

    docs, titles = ret

    return docs, titles


def train_lda_model(docs, topics=20, alpha='auto', beta='auto', no_above=0.5,
                    no_below=20, seed=None):
    """
    Trains the lda model

    Each parameter has the default value used also as default by the lda.py script
    :param docs: documents train the model upon
    :type docs: list[list[str]]
    :param topics: number of topics
    :type topics: int
    :param alpha: alpha parameter
    :type alpha: float or str
    :param beta: beta parameter
    :type beta: float or str
    :param no_above: keep terms which are contained in no more than this
        fraction of documents (fraction of total corpus size, not an absolute
        number).
    :type no_above: float
    :param no_below: keep tokens which are contained in at least this number of
        documents (absolute number).
    :type no_below: int
    :param seed: seed used for random generator
    :type seed: int or None
    :return: the trained model and the dictionary object used in training
    :rtype: tuple[LdaModel, Dictionary]
    """
    # Make a index to word dictionary.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below,
                               no_above=no_above)
    try:
        _ = dictionary[0]  # This is only to "load" the dictionary.
    except KeyError:
        sys.exit('Filtered documents are empty. Check the filter parameters '
                 'and the input files.')

    id2word = dictionary.id2token
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # Train LDA model.
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=len(corpus),
        alpha=alpha,
        eta=beta,
        num_topics=topics,
        random_state=seed
    )
    return model, dictionary


def prepare_topics(model, docs, titles, dictionary):
    """
    Prepare the dicts for the topics and the document topic assignment

    :param model: the trained lda model
    :type model: LdaModel
    :param docs: the documents to evaluate to assign the topics
    :type docs: list[list[str]]
    :param titles: the titles of the documents
    :type titles: list[str]
    :param dictionary: the gensim dictionary object used for training
    :type dictionary: Dictionary
    :return: the dict of the topics, the docs-topics assignement and
        the average coherence score
    :rtype: tuple[dict[int, dict[str, str or dict[str, float]]],
        list[dict[str, int or str or dict[int, str]]], float]
    """
    cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary,
                        coherence='c_v', processes=PHYSICAL_CPUS)

    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    avg_topic_coherence = cm.get_coherence()
    coherence = cm.get_coherence_per_topic()
    topics = {}
    topics_order = list(range(model.num_topics))
    topics_order.sort(key=lambda x: coherence[x], reverse=True)
    for i in topics_order:
        topic = model.show_topic(i)
        t_dict = {
            'name': f'Topic {i}',
            'terms_probability': {t[0]: float(t[1]) for t in topic},
            'coherence': f'{float(coherence[i]):.5f}',
        }
        topics[i] = t_dict

    docs_topics = []
    for i, (title, d) in enumerate(zip(titles, docs)):
        bow = dictionary.doc2bow(d)
        t = model.get_document_topics(bow)
        t.sort(key=lambda x: x[1], reverse=True)
        d_t = {
            'id': i,
            'title': title,
            'topics': {tu[0]: float(tu[1]) for tu in t},
        }
        docs_topics.append(d_t)

    return topics, docs_topics, avg_topic_coherence


def filter_placeholders(doc: str, placeholder, relevant_prefix):
    words = []
    for word in doc.split(' '):
        if word == placeholder:
            continue
        if word.startswith(relevant_prefix) and word.endswith(relevant_prefix):
            words.append(word.strip(relevant_prefix))
        else:
            words.append(word)

    return ' '.join(words)


def load_documents(preproc_file, target_col, title_col, delimiter,
                   placeholder=STOPWORD_PLACEHOLDER,
                   relevant_prefix=RELEVANT_PREFIX):
    dataset = pd.read_csv(preproc_file, delimiter=delimiter, encoding='utf-8')
    dataset.fillna('', inplace=True)
    with Pool(processes=PHYSICAL_CPUS) as pool:
        documents = pool.starmap(filter_placeholders,
                                 zip(dataset[target_col].to_list(),
                                     repeat(placeholder),
                                     repeat(relevant_prefix)))

    titles = dataset[title_col].to_list()
    return documents, titles


def output_topics(model, dictionary, docs, titles, outdir, file_prefix):
    topics, docs_topics, avg_topic_coherence = prepare_topics(model, docs,
                                                              titles,
                                                              dictionary)
    now = datetime.now()
    name = f'{file_prefix}_terms-topics_{now:%Y-%m-%d_%H%M%S}.json'
    topic_file = outdir / name
    with open(topic_file, 'w') as file:
        json.dump(topics, file, indent='\t')

    name = f'{file_prefix}_docs-topics_{now:%Y-%m-%d_%H%M%S}.json'
    docs_file = outdir / name
    with open(docs_file, 'w') as file:
        json.dump(docs_topics, file, indent='\t')


def load_acronyms(args):
    acro_df = pd.read_csv(args.acronyms, delimiter='\t', encoding='utf-8')
    assert_column(args.acronyms, acro_df, ['term', 'label'])
    acronyms = []
    for _, row in acro_df.iterrows():
        if row['label'] not in ['relevant', 'keyword']:
            continue

        sp = row['term'].split('|')
        acronyms.append(sp[1].strip(' ()').lower())
    del acro_df
    return acronyms


def lda(args):
    terms_file = args.terms_file
    preproc_file = args.preproc_file
    output_dir = args.outdir

    placeholder = args.placeholder
    relevant_prefix = placeholder

    if args.no_relevant:
        labels = ('keyword',)
    else:
        labels = ('keyword', 'relevant')

    additional_keyword = set()

    if args.additional_file is not None:
        for sfile in args.additional_file:
            additional_keyword |= load_additional_terms(sfile)

    if args.acronyms is not None:
        acronyms = load_acronyms(args)
    else:
        acronyms = None

    docs, titles = prepare_documents(preproc_file, terms_file,
                                     not args.no_ngrams, labels,
                                     args.target_column, args.title,
                                     delimiter=args.delimiter,
                                     additional_keyword=additional_keyword,
                                     acronyms=acronyms, placeholder=placeholder,
                                     relevant_prefix=relevant_prefix)

    if args.load_model is not None:
        lda_path = Path(args.load_model)
        model = LdaModel.load(str(lda_path / 'model'))
        dictionary = Dictionary.load(str(lda_path / 'model_dictionary'))
    else:
        no_below = args.no_below
        no_above = args.no_above
        topics = args.topics
        try:
            alpha = float(args.alpha)
        except ValueError:
            alpha = args.alpha
        try:
            beta = float(args.beta)
        except ValueError:
            beta = args.beta
        seed = args.seed
        model, dictionary = train_lda_model(docs, topics, alpha, beta,
                                            no_above, no_below, seed)

    topics, docs_topics, avg_topic_coherence = prepare_topics(model, docs,
                                                              titles,
                                                              dictionary)

    print(f'Average topic coherence: {avg_topic_coherence:.4f}.')
    output_topics(model, dictionary, docs, titles, output_dir, 'lda')

    if args.model:
        lda_path: Path = args.outdir / 'lda_model'
        lda_path.mkdir(exist_ok=True)
        model.save(str(lda_path / 'model'))
        dictionary.save(str(lda_path / 'model_dictionary'))


def main():
    args = init_argparser().parse_args()
    lda(args)


if __name__ == '__main__':
    main()
