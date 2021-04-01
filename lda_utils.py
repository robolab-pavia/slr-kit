from multiprocessing import Pool

import pandas as pd
from psutil import cpu_count

BARRIER_PLACEHOLDER = 'XXX'
RELEVANT_PREFIX = BARRIER_PLACEHOLDER
PHYSICAL_CPUS = cpu_count(logical=False)


def filter_barriers(doc):
    words = []
    for word in doc.split(' '):
        if word == BARRIER_PLACEHOLDER:
            continue
        if word.startswith(RELEVANT_PREFIX):
            words.extend(word.split('_')[1:])
        else:
            words.append(word)

    return ' '.join(words)


def load_documents(preproc_file, target_col):
    dataset = pd.read_csv(preproc_file, delimiter='\t', encoding='utf-8')
    dataset.fillna('', inplace=True)
    with Pool(processes=PHYSICAL_CPUS) as pool:
        documents = pool.map(filter_barriers, dataset[target_col].to_list())

    titles = dataset['title'].to_list()
    return documents, titles