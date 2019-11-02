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
    parser.add_argument('--n-grams', '-n', metavar='N', dest='n_grams', default=4,
                        help='maximum size of n-grams number')
    parser.add_argument('--num-n-grams', '-m', metavar='N', dest='num_n_grams', default=5000,
                        help='number of n-grams items')
    return parser


# Most frequently occuring n-grams
def get_top_n_grams(corpus, n=1, amount=10):
    if n > 1:
        vec = CountVectorizer(ngram_range=(n,n), max_features=amount).fit(corpus)
    else:
        vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:amount]


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


def load_stop_words(input_file, language='english'):
    stop_words_list = []
    with open(input_file, "r") as f:
        stop_words_list = f.read().split('\n')
    stop_words_list = [w for w in stop_words_list if w != '']
    stop_words_list = [w for w in stop_words_list if w[0] != '#']
    # Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))
    # Creating a list of custom stopwords
    new_words = stop_words_list
    stop_words = stop_words.union(new_words)
    return list(stop_words)


def process_corpus(dataset, stop_words):
    corpus = []
    for item in dataset:
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', item)
        # Convert to lowercase
        text = text.lower()
        # Remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
        # Remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ", text)
        # Convert to list from string
        text = text.split()
        # Stemming
        ps=PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words] 
        text = " ".join(text)
        corpus.append(text)
    return corpus


def main():
    parser = init_argparser()
    args = parser.parse_args()

    # TODO: create a function for these checkings

    # set the value of n_grams, possibly from the command line
    if args.n_grams is not None:
        try:
            n_grams = int(args.n_grams)
        except:
            print('Invalid value for parameter "n-grams": "{}"'.format(args.n_grams))
            sys.exit(1)
    else:
        n_grams = 4

    # set the value of num_n_grams, possibly from the command line
    if args.num_n_grams is not None:
        try:
            num_n_grams = int(args.num_n_grams)
        except:
            print('Invalid value for parameter "num_n-grams": "{}"'.format(args.num_n_grams))
            sys.exit(1)
    else:
        num_n_grams = 5000

    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)
    # TODO: write log string with values of the parameters used in the execution

    # load the dataset
    dataset = pandas.read_csv(args.datafile, delimiter = '\t')
    # TODO: check that a column called 'abstract1' actually exists in the input file
    debug_logger.debug("Dataset loaded {} items".format(len(dataset['abstract1'])))
    #logging.debug(dataset.head())

    if args.stop_words_file is not None:
        stop_words = load_stop_words(args.stop_words_file, language='english')
        debug_logger.debug("Stopwords loaded and updated")
    else:
        stop_words = []
    # print(stop_words)

    corpus = process_corpus(dataset['abstract1'], stop_words)
    debug_logger.debug("Corpus processed")

    # View corpus item
    # logging.debug(corpus[2])

    #cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,4))
    #X=cv.fit_transform(corpus)
    #l = list(cv.vocabulary_.keys())
    #print(l)
    #print(len(l))

    top_terms = get_top_n_words(corpus, n=None)
    all_terms = top_terms

    for n in range(2, n_grams + 1):
        top_terms = get_top_n_grams(corpus, n=n, amount=num_n_grams)
        all_terms.extend(top_terms)

    # write to output, either a file or stdout (default)
    output_file = open(args.output, 'w') if args.output is not None else sys.stdout
    writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['keyword', 'count', 'label'])
    for item in all_terms:
        writer.writerow([item[0], item[1], ''])
    if output_file is not sys.stdout:
        output_file.close()

    # # Fetch wordcount for each abstract
    # dataset['word_count'] = dataset['abstract1'].apply(lambda x: len(str(x).split(" ")))
    # print(dataset[['id','word_count']].head())
    # 
    # # Descriptive statistics of word counts
    # print(dataset.word_count.describe())
    # 
    # # Identify common words
    # freq = pandas.Series(' '.join(dataset['abstract1']).split()).value_counts()[:20]
    # print(freq)
    # 
    # #Identify uncommon words
    # freq1 =  pandas.Series(' '.join(dataset['abstract1']).split()).value_counts()[-20:]
    # print(freq1)


if __name__ == "__main__":
    main()
