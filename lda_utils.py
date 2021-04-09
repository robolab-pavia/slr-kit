from itertools import repeat
from multiprocessing import Pool

import pandas as pd
from psutil import cpu_count

from utils import BARRIER_PLACEHOLDER, RELEVANT_PREFIX

PHYSICAL_CPUS = cpu_count(logical=False)


def filter_barriers(doc, barrier_placeholder, relevant_prefix):
    words = []
    for word in doc.split(' '):
        if word == barrier_placeholder:
            continue
        if word.startswith(relevant_prefix):
            words.extend(word.split('_')[1:])
        else:
            words.append(word)

    return ' '.join(words)


def load_documents(preproc_file, target_col,
                   barrier_placeholder=BARRIER_PLACEHOLDER,
                   relevant_prefix=RELEVANT_PREFIX):
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    with Pool(processes=PHYSICAL_CPUS) as pool:
        documents = pool.starmap(filter_barriers,
                                 zip(dataset[target_col].to_list(),
                                     repeat(barrier_placeholder),
                                     repeat(relevant_prefix)))

    titles = dataset['title'].to_list()
    return documents, titles