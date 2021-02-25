r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import io
import sys
import os.path
import re
import tarfile
import pandas

import smart_open


def generate_raw_docs():
    dataset = pandas.read_csv('rts_preproc.csv', delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    docs = [d.split(' ') for d in documents]
    return docs

def generate_filtered_docs():
    words_dataset = pandas.read_csv('dataset/rts/tags_facchinetti/rts_terms_tab.csv', delimiter='\t', encoding='utf-8')
    terms = words_dataset['keyword'].to_list()
    labels = words_dataset['label'].to_list()
    zipped = zip(terms, labels)
    good = [x for x in zipped if x[1] == 'keyword' or x[1] == 'relevant']
    good_set = set()
    for x in good:
        good_set.add(x[0])
    #print(good_set)
    dataset = pandas.read_csv('rts_preproc.csv', delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    documents = dataset['abstract_lem'].to_list()
    docs = [d.split(' ') for d in documents]

    good_docs = []
    for doc in docs:
        gd = [t for t in doc if t in good_set]
        good_docs.append(gd)
    return good_docs

good_docs = generate_filtered_docs()
#good_docs = generate_raw_docs()
#good_docs = good_docs[:2000]
docs = good_docs
#sys.exit(1)

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
if False:
    num_topics = 20
    chunksize = 4000
    passes = 20
    iterations = 600
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
else:
    num_topics = 30
    chunksize = 2000
    passes = 20
    iterations = 4
    eval_every = 5  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

for d in docs:
    bow = dictionary.doc2bow(d)
    t = model.get_document_topics(bow)
    print(t)
