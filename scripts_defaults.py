from utils import BARRIER_PLACEHOLDER

defaults = {
    'preprocess': {
        'datafile': {'value': '', 'required': True},
        'output': {'value': '-', 'required': False},
        'placeholder': {'value': BARRIER_PLACEHOLDER, 'required': False},
        'barrier-words': {'value': [], 'required': False,
                          'dest': 'barrier_words_file'},
        'relevant-terms': {'value': [[]], 'required': False,
                           'non-standard': True, 'dest': 'relevant_terms_file'},
        'acronyms': {'value': '', 'required': False},
        'target-column': {'value': 'abstract', 'required': False},
        'output-column': {'value': 'abstract_lem', 'required': False},
        'input-delimiter': {'value': '\t', 'required': False},
        'output-delimiter': {'value': '\t', 'required': False},
        'language': {'value': 'en', 'required': False},
        'regex': {'value': '', 'required': False},
        'rows': {'value': '', 'required': False, 'dest': 'input_rows'},
    },
    'gen_terms': {
        'datafile': {'value': '', 'required': True},
        'output': {'value': '', 'required': True},
        'stdout': {'value': False, 'required': False},
        'n-grams': {'value': 4, 'required': False},
        'min-frequency': {'value': 5, 'required': False},
        'placeholder': {'value': BARRIER_PLACEHOLDER, 'required': False},
        'column': {'value': 'abstract_lem', 'required': False},
    },
    'lda': {
        'preproc_file': {'value': '', 'required': True},
        'terms_file': {'value': '', 'required': True},
        'outdir': {'value': '', 'required': False, 'non-standard': True},
        'text-column': {'value': 'abstract_lem', 'required': False,
                        'dest': 'target_column'},
        'title-column': {'value': 'title', 'required': False,
                         'dest': 'title'},
        'additional-terms': {'value': [], 'required': False,
                             'dest': 'additional_file'},
        'acronyms': {'value': '', 'required': False},
        'topics': {'value': 20, 'required': False},
        'alpha': {'value': 'auto', 'required': False},
        'beta': {'value': 'auto', 'required': False},
        'no_below': {'value': 20, 'required': False},
        'no_above': {'value': 0.5, 'required': False},
        'seed': {'value': '', 'required': False},
        'ngrams': {'value': False, 'required': False},
        'model': {'value': False, 'required': False},
        'no-relevant': {'value': False, 'required': False},
        'placeholder': {'value': BARRIER_PLACEHOLDER, 'required': False},
        'load-model': {'value': '', 'required': False}
    },
    'lda_grid_search': {
        'preproc_file': {'value': '', 'required': True},
        'terms_file': {'value': '', 'required': True},
        'outdir': {'value': '', 'required': False, 'non-standard': True},
        'text-column': {'value': 'abstract_lem', 'required': False,
                        'dest': 'target_column'},
        'title-column': {'value': 'title', 'required': False,
                         'dest': 'title'},
        'additional-terms': {'value': [], 'required': False,
                             'dest': 'additional_file'},
        'acronyms': {'value': [], 'required': False},
        'min-topics': {'value': 5, 'required': False},
        'max-topics': {'value': 20, 'required': False},
        'step-topics': {'value': 1, 'required': False},
        'result': {'value': '-', 'required': False, 'non-standard': True},
        'seed': {'value': '', 'required': False},
        'ngrams': {'value': False, 'required': False},
        'model': {'value': False, 'required': False},
        'no-relevant': {'value': False, 'required': False},
        'placeholder': {'value': BARRIER_PLACEHOLDER, 'required': False},
        'plot-show': {'value': False, 'required': False},
        'plot-save': {'value': False, 'required': False},
    }
}
# create a dict with the default value of each parameter for each script
scripts_defaults = {script: {k: v['value'] for k, v in default.items()}
                    for script, default in defaults.items()}

PREPROCESS_DEFAULTS = scripts_defaults['preprocess']
GENTERMS_DEFAULTS = scripts_defaults['gen_terms']
LDA_DEFAULTS = scripts_defaults['lda']
LDAGRIDSEARCH_DEFAULTS = scripts_defaults['lda_grid_search']

