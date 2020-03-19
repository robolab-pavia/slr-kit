import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
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
    parser.add_argument('--plot', '-p', metavar='FILENAME',
                        help='output file name for coherence plot')
    parser.add_argument('--topics-words-matrix', '-twm', metavar='FILENAME', dest='topics_words_matrix',
                        help='output file name with (topics x words) matrix in CSV format')
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name with (documents x topics) matrix in CSV format')
    return parser


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):

        # LDA multi-core implementation with maximum number of workers
        model = models.LdaMulticore(corpus,
                                    num_topics=num_topics,
                                    id2word=dictionary,
                                    chunksize=1000,
                                    passes=100,
                                    random_state=100,
                                    minimum_probability=0.0)
        model_list.append(model)

        # computes coherence score for that model
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def get_vocabulary(occurences, dictionary):
    """
    Iterate over documents-terms matrix to construct a Gensim dictionary
    from only classified keywords
    :param occurences: vector as BoW for a document (id,occurences)
    :param dictionary: custom dictionary based on classification (FA.WO.C)
    :return:
    """
    doc = occurences.values.tolist()  # convert row to list
    ngrams = []
    for i in range(len(doc)):  # for each element
        if doc[i] != 0:
            ngrams.append(dictionary.iloc[i, 1])  # match from vocabulary the term with positional index
    return ngrams


def main():
    debug_logger.debug('[multidimensional scaling] Started')
    parser = init_argparser()
    args = parser.parse_args()

    tdm = pd.read_csv(args.infile, delimiter='\t', index_col=0)
    tdm.fillna('', inplace=True)

    # we want to use the documents-terms matrix so
    dtm = tdm.T

    # TODO: allow the selection of the filename from command line
    terms = load_df('term-list.csv', required_columns=['id', 'term'])

    debug_logger.debug('[multidimensional scaling] Computing LDA')

    documents = []
    i = 0
    for index, row in dtm.iterrows():
        doc = get_vocabulary(row, terms)
        documents.append(doc)
        i = i + 1
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Checks for pre-trained model file named 'model.bin'
    # if model exists than loads it, otherwise starts training

    if os.path.isfile('model.bin'):
        lda = models.LdaModel.load('model.bin')
    else:

        start = 5
        limit = 50
        step = 5

        # Iterate LDA training on multiple models with different num_topics
        # in range start:limit:step
        model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                                corpus=corpus,
                                                                texts=documents,
                                                                start=start,
                                                                limit=limit,
                                                                step=step)
        # Show graph for coherence
        x = range(start, limit, step)

        if args.plot:
            plt.plot(x, coherence_values, color='b',
                     marker='o', linestyle='solid',
                     markerfacecolor='none',
                     markeredgecolor='b')

            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.legend("coherence_values", loc='best')
            plt.savefig(args.plot, dpi=300)
            plt.show()

        # Optimal model selection based on iterated test taking as best model
        # the one that maximises the coherence score

        num_topics = start
        lda = []
        for m, model, cv in zip(x, model_list, coherence_values):
            if cv == max(coherence_values):
                num_topics = m
                lda = model  # select optimum model

        lda.save("model.bin")

    # Computing topic distribution over documents
    # creates a ( documents x topics ) matrix

    docs = []
    for i, row in enumerate(lda[corpus]):
        doc = []
        row = sorted(row, key=lambda x: (x[0]), reverse=False)
        for j, (topic_num, prop_topic) in enumerate(row):
            doc.append(prop_topic)  # append doc
            # print("Doc: %d - Topic: %d - P: %f" % (i, topic_num, prop_topic))
        docs.append(doc)

    # Computing features distribution over topics
    # creates a ( topics x words ) matrix

    topics_matrix = lda.show_topics(num_topics=15, formatted=False, num_words=10)
    topics_matrix = sorted(topics_matrix, key=lambda x: (x[0]), reverse=False)

    topics = []
    topic_num = []
    for i, row in enumerate(topics_matrix):
        topic_num.append(row[0])
        words = []
        for index, (word,score) in enumerate(row[1]):
            words.append(word)
        topics.append(words)

    debug_logger.debug('[multidimensional scaling] Saving')

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(topics)
    df_topic_keywords.insert(0, "topic", topic_num, True)

    if not args.topics_words_matrix:
        print(df_topic_keywords.to_string())
    else:
        output_file = open(args.topics_words_matrix, 'w', encoding='utf-8', newline='')
        export_csv = df_topic_keywords.to_csv(output_file, header=True, sep='\t')
        output_file.close()

    df = pd.DataFrame(docs, index=dtm.index)
    df.fillna(0, inplace=True)

    if not args.output:
        print(df.to_string())
    else:
        output_file = open(args.output, 'w', encoding='utf-8', newline='')
        export_csv = df.to_csv(output_file, header=True, sep='\t', float_format='%.6f')
        output_file.close()

    debug_logger.debug('[multidimensional scaling] Terminated')


if __name__ == "__main__":
    main()
