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
    ##Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))
    ##Creating a list of custom stopwords
    new_words = stop_words_list
    stop_words = stop_words.union(new_words)
    return stop_words


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


def init_logger(logfile):
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


logging.basicConfig(filename='slr-kit.log', filemode='a', format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

# load the dataset
dataset = pandas.read_csv(sys.argv[1], delimiter = '\t')
logging.debug("Dataset loaded {} items".format(len(dataset['abstract1'])))
#logging.debug(dataset.head())

stop_words = load_stop_words("stop_words.txt", language='english')
logging.debug("Stopword loaded and updated")

corpus = process_corpus(dataset['abstract1'], stop_words)
logging.debug("Corpus processed")

# View corpus item
# logging.debug(corpus[2])

#cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,4))
#X=cv.fit_transform(corpus)
#l = list(cv.vocabulary_.keys())
#print(l)
#print(len(l))

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=None)
all_words = top_words

top2_words = get_top_n_grams(corpus, n=2, amount=5000)
all_words.extend(top2_words)

top3_words = get_top_n_grams(corpus, n=3, amount=5000)
all_words.extend(top3_words)

top4_words = get_top_n_grams(corpus, n=4, amount=5000)
all_words.extend(top4_words)

with open('output_x.csv', mode='w') as output_file:
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['keyword', 'count', 'group'])
    for item in all_words:
        writer.writerow([item[0], item[1], ''])

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
