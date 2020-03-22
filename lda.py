import argparse
import glob
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from gensim import corpora, models
from gensim.models import CoherenceModel

from utils import (
    load_df,
    setup_logger,
)

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser(description='Perform MDS over a distance matrix')
    parser.add_argument('infile', action="store", type=str,
                        help="input CSV file with documents-terms matrix")
    parser.add_argument('-DM', action='store_true',
                        help='Delete pre-trained model for new training')
    parser.add_argument('--min-topics', '-m', type=int, dest='m', metavar='5',
                        help='Minimum number of topics to retrieve (default: 5)')
    parser.add_argument('--max-topics', '-M',  type=int, dest='M', metavar='20',
                        help='Maximum number of topics to retrieve (default: 20)')
    parser.add_argument('--step-topics', '-s', type=int, dest='s', metavar='1',
                        help='Step in range(min,max,step) for topics retrieving (default: 1)')
    parser.add_argument('--plot', '-p', metavar='FILENAME',
                        help='output file name for coherence plot')
    parser.add_argument('--topics-terms-matrix', '-twm', metavar='FILENAME', dest='topics_terms_matrix',
                        help='output file name with (topics x terms) matrix in CSV format')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name with (documents x topics) matrix in CSV format')
    return parser


def compute_optimal_model(dictionary, corpus, texts, limit, start, step, alpha, beta):
    """
    Train several models iterating over the specified number of topics and performs
    LDA hyper-parameters alpha and beta tuning
    :param dictionary: Gensim dictionary from FAWOC classification
    :param corpus: Gensim corpus
    :param texts: Tokenized text, used for window sliding
    :param limit: Maximum number of topics
    :param start: Minimum number of topics
    :param step: Step for topics range
    :param alpha: Alpha parameter values to test
    :param beta: Beta parameter values to test
    :return: pandas data-frame with best model performances
    """
    # Topics range
    topics_range = range(start, limit, step)

    # Validation sets
    num_of_docs = len(corpus)

    # corpus must be the last element!
    corpus_sets = [corpus]  # corpus_sets[ClippedCorpus(corpus, int(num_of_docs * 0.75)), corpus]

    # 'Full' must be the last element!
    corpus_title = ['Full']  # corpus_title = ['Clipped', 'Full']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }
    # Can take a long time to run
    total_iterations = len(corpus_sets) * len(topics_range) * len(alpha) * len(beta)

    progress_bar = tqdm.tqdm(total=total_iterations)

    # iterate through validation corpus
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterate through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i],
                                                  dictionary=dictionary,
                                                  texts=texts,
                                                  num_topics=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    progress_bar.update(1)

    df = pd.DataFrame(model_results)
    # df.to_csv('lda_tuning_results.csv', sep='\t', index=False)
    progress_bar.close()

    return df


def compute_coherence_values(dictionary, corpus, texts, num_topics, a, b):
    """
    Compute c_v coherence for various number of topics

    :param dictionary: Gensim dictionary from FAWOC classification
    :param corpus: Gensim corpus
    :param texts: Tokenized text, used for window sliding
    :param num_topics: Number of topics
    :param a: Alpha parameter
    :param b: Eta parameter
    :return: Coherence value corresponding to the LDA model
    """
    model_list = []

    # LDA multi-core implementation with maximum number of workers
    model = models.LdaMulticore(corpus,
                                num_topics=num_topics,
                                id2word=dictionary,
                                chunksize=100,
                                passes=10,
                                random_state=100,
                                minimum_probability=0.0,  # important to match dimensions
                                alpha=a,
                                eta=b)
    model_list.append(model)

    # computes coherence score for that model
    cv_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return cv_model.get_coherence()


def get_vocabulary(bow):
    """
    Iterate over documents-terms matrix to construct a Gensim dictionary
    from only classified keywords
    :param bow: vector as BoW for a document (keyword_id, counts)
    :return: List of all keywords contained into the document
    """
    # TODO: allow the selection of the filename from command line
    # load the dictionary of keywords [ keyword_id - term ]
    dictionary = load_df('term-list.csv', required_columns=['id', 'term'])
    doc = bow.values.tolist()  # convert row to list
    n_grams = []
    for i in range(len(doc)):  # for each element
        if doc[i] != 0:
            n_grams.append(dictionary.iloc[i, 1])  # match from vocabulary the term with positional index
    return n_grams


def main():
    debug_logger.debug('[Latent Dirichlet allocation] Started')
    parser = init_argparser()
    args = parser.parse_args()

    if args.DM:  # user asks to delete the pre-trained model
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Running with -DM will permanently delete pre-trained model."
                           "Are you sure? [Y/N]? ").lower()
            if answer == "y":
                # Handle errors while calling os.remove()
                try:
                    [os.remove(x) for x in glob.glob("model.bin*")]
                except OSError:
                    print("Error while deleting pre-trained model")
            else:
                print("Aborting")
                exit()
        return answer == "y"

    tdm = pd.read_csv(args.infile, delimiter='\t')
    tdm.fillna('', inplace=True)

    # we want to use the documents-terms matrix so we transpose td-matrix
    # then a workaround to re-set docs ids as correct data-frame index
    dtm = tdm.T
    dtm = dtm.rename(columns=dtm.iloc[0]).drop(dtm.index[0])
    dtm.index.name = None  # drop 'Unnamed: 0' coming from transposition

    debug_logger.debug('[Latent Dirichlet allocation] Computing LDA')

    documents = []
    docs_ids = []  # preserve documents IDs (due to some empties abstracts)
    for index, row in dtm.iterrows():
        doc = get_vocabulary(row)
        docs_ids.append(index)
        documents.append(doc)
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Checks for pre-trained model file named 'model.bin'
    # if model exists than loads it, otherwise starts training
    if os.path.isfile('model.bin'):
        lda = models.LdaModel.load('model.bin')
        opt_topics_num = lda.num_topics
    else:

        # Range of topics number to test from arguments or defaults:
        start = 5
        limit = 21  # + 1
        step = 1
        if args.m:
            start = int(args.m)
        if args.M:
            limit = int(args.M) + 1  # since we are using range(start,limit,step)
        if args.s:
            step = int(args.s)

        # Alpha parameter
        alpha = list(np.arange(0.01, 1, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')
        # Beta parameter
        beta = list(np.arange(0.01, 1, 0.3))
        beta.append('symmetric')

        debug_logger.debug('[Latent Dirichlet allocation] LDA hyper-parameters tuning')
        # Optimal model selection based on iterated test taking as best model
        # the one that maximises the coherence score.
        # Iterates LDA training on multiple models with different num_topics
        # in range start:limit:step
        evaluation_df = compute_optimal_model(dictionary=dictionary,
                                              corpus=corpus,
                                              texts=documents,
                                              start=start,
                                              limit=limit,
                                              step=step,
                                              alpha=alpha,
                                              beta=beta)

        # In case of multiple validation sets extract results only for full corpus
        corpus_100 = evaluation_df[evaluation_df['Validation_Set'] == 'Full']
        topics_num = evaluation_df.iloc[[corpus_100['Coherence'].idxmax()]]['Topics'].values[0]
        opt_model = corpus_100[corpus_100['Topics'] == topics_num]

        # Extract optimum model parameters values
        opt_topics_num = topics_num
        opt_alpha = evaluation_df.iloc[[corpus_100['Coherence'].idxmax()]]['Alpha'].values[0]
        opt_beta = evaluation_df.iloc[[corpus_100['Coherence'].idxmax()]]['Beta'].values[0]
        opt_c_v = evaluation_df.iloc[[corpus_100['Coherence'].idxmax()]]['Coherence'].values[0]

        # train the optimum model
        lda = models.LdaMulticore(corpus,
                                  num_topics=opt_topics_num,
                                  id2word=dictionary,
                                  chunksize=100,
                                  passes=10,
                                  random_state=100,
                                  minimum_probability=0.0,
                                  alpha=opt_alpha,
                                  eta=opt_beta)
        lda.save("model.bin")

        if args.plot:
            # Show graph for coherence and reshape data-frame fot topics-c_v plot
            corpus_100.insert(1, 'temp', np.max(corpus_100.Coherence.values, axis=0))
            idx = corpus_100.groupby('Topics')['temp'].idxmax()  # get max score per topics number
            data = corpus_100.loc[idx].drop('temp', 1)  # drops temporary column

            x = range(start, limit, step)
            plt.plot(x, data['Coherence'], color='b',
                     marker='.', linestyle='solid',
                     markerfacecolor='w',
                     markeredgecolor='b')
            plt.xlabel("Number of Topics")
            plt.ylabel("Coherence score")
            plt.legend("coherence_values", loc='best')
            plt.savefig(args.plot, dpi=300)
            plt.show()

    # Computing topic distribution and contributions over documents
    # creates a ( documents x topics ) matrix
    docs = []
    dominant_topics = []
    for i, row in enumerate(lda[corpus]):
        doc = []
        row = sorted(row, key=lambda itm: (itm[0]), reverse=False)
        for j, (topic_num, prop_topic) in enumerate(row):
            doc.append(prop_topic)  # append doc
            # print("Doc: %d - Topic: %d - P: %f" % (i, topic_num, prop_topic))
        dominant_topics.append(doc.index(max(doc)))
        docs.append(doc)

    # Computing features distribution over topics
    # creates a ( topics x words ) matrix
    topics_matrix = lda.show_topics(num_topics=opt_topics_num, formatted=False, num_words=10)
    topics_matrix = sorted(topics_matrix, key=lambda itm: (itm[0]), reverse=False)

    topics = []
    topic_num = []
    for i, row in enumerate(topics_matrix):
        topic_num.append(row[0])
        words = []
        for index, (word, score) in enumerate(row[1]):
            words.append(word)
        topics.append(words)

    debug_logger.debug('[Latent Dirichlet allocation] Saving Documents-Topics matrix')

    # Documents-topics matrix data-frame for output
    df = pd.DataFrame(docs, index=docs_ids)
    df.insert(0, "dominant_topic", dominant_topics, True)
    df.fillna(0, inplace=True)

    if not args.output:
        print("Documents-Topics Matrix")
        print(df.to_string())
    else:
        output_file = open(args.output, 'w', encoding='utf-8', newline='')
        export_csv = df.to_csv(output_file, header=True, sep='\t', float_format='%.6f')
        output_file.close()

    # get best document for each topic
    # max_values_indexes = df.loc[:, df.columns != 'dominant_topic'].idxmax()
    # topic_index = 0
    # best_documents = []
    # for row in max_values_indexes:
    #     best_documents.append(row)
    #     topic_index = topic_index + 1

    best_documents = []
    N = 3  # TODO: maybe from commandline?
    for column in df.loc[:, df.columns != 'dominant_topic']:
        sorted_df = df.sort_values(by=[column], ascending=False)
        top_n = []
        i = 0
        for index, row in sorted_df.iterrows():
            if i < N:  # get only top N
                top_n.append(index)
                i = i + 1
            else:
                break

        best_documents.append(top_n)

    debug_logger.debug('[Latent Dirichlet allocation] Saving Topics-Keywords matrix')
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(topics)
    df_topic_keywords.insert(0, "topic", topic_num, True)
    df_topic_keywords.insert(1, "best_doc", best_documents, True)

    if not args.topics_terms_matrix:
        print("Topics-Terms Matrix")
        print(df_topic_keywords.to_string())
    else:
        output_file = open(args.topics_terms_matrix, 'w', encoding='utf-8', newline='')
        export_csv = df_topic_keywords.to_csv(output_file, header=True, sep='\t')
        output_file.close()

    debug_logger.debug('[Latent Dirichlet allocation] Terminated')


if __name__ == "__main__":
    main()
