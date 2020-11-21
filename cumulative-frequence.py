import pandas
# Libraries for text preprocessing
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
import csv
import logging
import argparse
from utils import setup_logger
import numpy as np
import matplotlib.pyplot as plt


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--showfig',
                        help="plots the chart",
                        action='store_true',
                        dest='showfig', default=False)
    parser.add_argument('--savefig',
                        help="plots the chart",
                        action='store_true',
                        dest='savefig', default=False)
    parser.add_argument('--output',
                        help="saves the csv file",
                        action='store_true',
                        dest='output', default=False)
    parser.add_argument('--n-grams', '-n', metavar='N', dest='n_grams', default=4,
                        help='maximum size of n-grams number')
    parser.add_argument('--top',
                        help='number of N-grams to display',
                        dest='top',
                        type=int,
                        default=10)
    parser.add_argument('--num-n-grams', '-m', metavar='N', dest='num_n_grams',
                        default=5000,
                        help='number of n-grams items')
    return parser


# Plots an heatmap of most frequency. TODO Rifattorizzare
def heatmap(corpi, n=1, amount=10):
    series = []
    if n > 1:
        corpi = [corpus for corpus in corpi if corpus != '']
        for idx, corpus in enumerate(corpi, start=1):
            vec = CountVectorizer(ngram_range=(n,n), max_features=amount).fit([corpus])
            bag_of_words = vec.transform([corpus])
            sum_words = bag_of_words.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

            y = [word[0] for word in words_freq]
            dfCorpus = pandas.Series([word[1] for word in words_freq], index=y, name=idx)
            series.append(dfCorpus)
        dfN = pandas.concat(series, axis=1, ignore_index=True)
        dfN = dfN.fillna(0)
        dfN = dfN.cumsum(axis=1)
        dfN = dfN.sort_values([idx-1], ascending=False)
        dfN = dfN.iloc[:idx, :]
        dfN.columns = range(1, idx+1)
        dfLog = np.log10(dfN).replace(-np.inf, 0)


        ax = sns.heatmap(dfN, annot=True)

        # plt.pcolor(dfN.sort_values([idx-1], ascending=True))
        # log
        # plt.pcolor(dfLog.sort_values([idx-1], ascending=True))
        plt.yticks(np.arange(0.5, len(dfLog.index), 1), dfLog.index)
        plt.xticks(np.arange(0.5, len(dfLog.columns), 1), dfLog.columns)
        plt.title("Somma cumulativa frequenza bigrammi in 815 articoli\n\
                    in scala logaritmica")
        plt.xlabel("Papers #")
        plt.ylabel("Bigrammi")
        # cbar = plt.colorbar()
        # cbar.ax.set_ylabel('Frequenza', rotation=270)
        plt.show()


        dfN.to_csv('dfN-cum.csv')

        return words_freq[:amount]
    else:
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

        return words_freq[:amount]


def charts(corpi, args, n=1, amount=10, top=10):
    series = []
    corpi = [corpus for corpus in corpi if corpus != '']
    for idx, corpus in enumerate(corpi, start=1):
        if n > 1:
            vec = CountVectorizer(ngram_range=(n,n), max_features=amount).fit([corpus])
        else:
            vec = CountVectorizer().fit([corpus])
        bag_of_words = vec.transform([corpus])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

        y = [word[0] for word in words_freq]
        # TODO forse idx ed enumrate non servono proprio
        serie = pandas.Series([word[1] for word in words_freq], index=y, name=idx)
        series.append(serie)
    dfN = pandas.concat(series, axis=1, ignore_index=True)
    dfN = dfN.fillna(0)
    dfN = dfN.T.cumsum(axis=0)
    # Used to sort based on last value row
    newSorting = dfN.columns[dfN.iloc[dfN.last_valid_index()].argsort()]
    dfN = dfN[newSorting]
    dfN.index.name = 'paper#'
    

    if args.showfig or args.savefig:
        dfN.iloc[:, -top:].plot()
        plt.title("Top {} {}-gram cumulative frequency in {} papers".format(top, n, len(corpi)))
        plt.xlabel("Papers #")
        plt.ylabel("Frequency")

        if args.showfig:
            plt.show()
        if args.savefig:
            os.makedirs("./n-grams_frequency", 0o775, exist_ok=True)
            plt.savefig("./n-grams_frequency/{}_{}-gram.png".format(''.join(args.datafile.split('.')[:-1]), n),
                        bbox_inches = "tight", dpi=256)
    return dfN




#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    #print("Length: {}".format(len(words_freq)))
    return words_freq[:n]


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
    num_n_grams = convert_int_parameter(args, 'num_n_grams', default=5000)

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

    corpus = dataset[target_column].to_list()

    top_terms = get_top_n_words(corpus, n=None)

    for n in range(1, n_grams + 1):
        dfN = charts(corpus, args, n=n, amount=num_n_grams, top=args.top)
        
        if args.output:
            os.makedirs("./n-grams_frequency", 0o775, exist_ok=True)
            dfN.to_csv("./n-grams_frequency/{}_{}-gram.csv".format(''.join(args.datafile.split('.')[:-1]), n))


if __name__ == "__main__":
    main()
