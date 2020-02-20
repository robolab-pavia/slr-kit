import pandas
# Libraries for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import sys
import csv
import logging
import argparse


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """Function to setup a generic loggers."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--output', '-o', metavar='FILENAME',
                        help='output file name')
    parser.add_argument('--stop-words', '-s', metavar='FILENAME', dest='stop_words_file',
                        help='stop words file name')
    return parser


def load_stop_words(input_file, language='english'):
    stop_words_list = []
    with open(input_file, "r", encoding='utf-8') as f:
        stop_words_list = f.read().split('\n')
    stop_words_list = [w for w in stop_words_list if w != '']
    stop_words_list = [w for w in stop_words_list if w[0] != '#']
    # Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))
    # Creating a list of custom stopwords
    new_words = stop_words_list
    stop_words = stop_words.union(new_words)
    return list(stop_words)


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def main():
    target_column = 'abstract_lem'
    parser = init_argparser()
    args = parser.parse_args()

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

    if args.stop_words_file is not None:
        stop_words = load_stop_words(args.stop_words_file, language='english')
        debug_logger.debug("Stopwords loaded and updated")
    else:
        stop_words = []
    # print(stop_words)

    #cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=10000)
    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 4))
    word_count_vector = cv.fit_transform(dataset[target_column])
    #print(list(cv.vocabulary_.keys())[:10])

    # only needed once, this is a mapping of index to 
    feature_names = cv.get_feature_names()

    tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)
    idf = tfidf_transformer.fit(word_count_vector)

    #generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform(dataset[target_column]))
    #print(tf_idf_vector)

    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    #print(sorted_items)
     
    # extract only the top n
    keywords = extract_topn_from_vector(feature_names, sorted_items, 5000)
    # TODO: needs sorting by td-idf
     
    # print the results
    print("Term\ttd-idf")
    for k in keywords:
        print('{}\t{}'.format(k, keywords[k]))

    # View corpus item
    # logging.debug(corpus[2])


if __name__ == "__main__":
    main()
