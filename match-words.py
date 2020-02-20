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
import enum
from dataclasses import dataclass

# List of class names
class WordClass(enum.Enum):
    """
    Class for a classified word.

    Each member contains a classname and a key.
    classname is a str. It should be a meaningful word describing the class.
    It goes in the csv as the word classification
    key is a str. It is the key used by the program to classify a word.

    In WordClass memeber creation the tuple is (classname, default_key)
    In the definition below the NONE WordClass is provided to 'classify' an
    un-marked word.
    """
    NONE = ('', '')
    KEYWORD = ('keyword', 'k')
    NOISE = ('noise', 'n')
    RELEVANT = ('relevant', 'r')
    NOT_RELEVANT = ('not-relevant', 'x')
    POSTPONED = ('postponed', 'p')

    @staticmethod
    def get_from_key(key: str):
        for clname in WordClass:
            if clname.key == key:
                return clname

        raise ValueError('"{}" is not a valid key'.format(key))

    @staticmethod
    def get_from_classname(classname):
        for clname in WordClass:
            if clname.classname == classname:
                return clname

        raise ValueError('"{}" is not a valid class name'.format(classname))

    def __init__(self, classname, key):
        self.classname = classname
        self.key = key


@dataclass
class Word:
    index: int
    word: str
    count: int
    group: WordClass
    order: int
    related: str

    def is_grouped(self):
        return self.group != WordClass.NONE


class WordList(object):
    def __init__(self, items=None):
        self.items = items
        self.csv_header = None

    def from_csv(self, infile):
        with open(infile, newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            header = csv_reader.fieldnames
            items = []
            for row in csv_reader:
                order_value = row['order']
                if order_value == '':
                    order = None
                else:
                    order = int(order_value)

                related = row.get('related', '')
                try:
                    group = WordClass.get_from_classname(row['label'])
                except ValueError:
                    group = WordClass.get_from_key(row['label'])

                item = Word(
                    index=0,
                    word=row['keyword'],
                    count=row['count'],
                    group=group,
                    order=order,
                    related=related
                )
                items.append(item)

        if 'related' not in header:
            header.append('related')

        self.csv_header = header
        self.items = items
        return header, items

    def to_csv(self, outfile):
        with open(outfile, mode='w', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=self.csv_header,
                                    delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for w in self.items:
                item = {'keyword': w.word,
                        'count': w.count,
                        'label': w.group.classname,
                        'order': w.order,
                        'related': w.related}
                writer.writerow(item)

    def get_last_inserted_order(self):
        orders = [w.order for w in self.items if w.order is not None]
        if len(orders) == 0:
            order = -1
        else:
            order = max(orders)

        return order

    def get_last_inserted_word(self):
        last = self.get_last_inserted_order()
        for w in self.items:
            if w.order == last:
                return w
        else:
            return None

    def mark_word(self, word, marker, order, related=''):
        for w in self.items:
            if w.word == word:
                w.group = marker
                w.order = order
                w.related = related
                break

        return self

    def return_related_items(self, key):
        containing = []
        not_containing = []
        for w in self.items:
            if w.is_grouped():
                continue

            if find_word(w.word, key):
                containing.append(w.word)
            else:
                not_containing.append(w.word)

        return containing, not_containing

    def count_classified(self):
        return len([item for item in self.items if item.is_grouped()])

    def count_by_class(self, cls):
        return len([w for w in self.items if w.group == cls])

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
    for item in dataset[:1]:
        print(item)
        print()
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', item)
        print(text)
        print()
        # Convert to lowercase
        text = text.lower()
        print(text)
        print()
        # Remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
        print(text)
        print()
        # Remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ", text)
        print(text)
        print()
        # Convert to list from string
        text = text.split()
        print(text)
        print()
        # Stemming
        ps=PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words] 
        text = " ".join(text)
        print(text)
        print()
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


def find_full_string(string1, string2):
   if re.search(r"\b" + re.escape(string1) + r"\b", string2):
      return True
   return False


def main():
    logging.basicConfig(filename='slr-kit.log', filemode='a', format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

    # load the dataset
    dataset = pandas.read_csv(sys.argv[1], delimiter = '\t')
    logging.debug("Dataset loaded {} items".format(len(dataset['abstract1'])))
    #logging.debug(dataset.head())

    stop_words = load_stop_words("stop_words.txt", language='english')
    logging.debug("Stopword loaded and updated")

    corpus = process_corpus(dataset['abstract1'], stop_words)
    logging.debug("Corpus processed")
    return

    words = WordList()
    _, _ = words.from_csv('words-rts-full.csv')
    not_relevant = [w for w in words.items if w.group == WordClass.NOT_RELEVANT]
    relevant = [w for w in words.items if w.group == WordClass.RELEVANT]

    summary = {}
    for i, abstract in enumerate(corpus):
        print('Processing abstract {}'.format(i))
        #for w in not_relevant:
        for w in relevant:
            word = w.word
            if find_full_string(word, abstract):
                if word not in summary:
                    summary[word] = [i]
                else:
                    summary[word].append(i)
    print(summary)
    all_papers = []
    for term in summary:
        all_papers.extend(summary[term])
    print(sorted(set(all_papers)))




    # View corpus item
    # logging.debug(corpus[2])
    # print(corpus[2])

    #cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,4))
    #X=cv.fit_transform(corpus)
    #l = list(cv.vocabulary_.keys())
    #print(l)
    #print(len(l))


if __name__ == "__main__":
    main()
