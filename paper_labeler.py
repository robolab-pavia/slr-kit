import pathlib

import pandas as pd
from RISparser import readris
from slrkit_utils.argument_parser import ArgParse


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    parser = ArgParse()

    parser.add_argument('ris_file', type=str, help='path to the ris file',
                        suggest_suffix='.ris')
    parser.add_argument('abstract_file', type=str,
                        help='path to the file with the abstracts of the papers',
                        input=True)
    parser.add_argument('journal_file', type=str,
                        help='path to the file with the classified journals',
                        input=True)

    return parser


def ris_reader(ris_path):
    """
    Creates a list of journals and papers titles from the ris file

    :param ris_path: path to the ris file
    :type ris_path: pathlib.Path
    :return: List of titles and relative journal
    :rtype: pd.DataFrame
    """
    paper_journal_list = {
        'title': [],
        'journal': [],
    }

    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            if 'title' not in entry:
                continue
            try:
                journal = entry['secondary_title']
            except KeyError:
                journal = entry.get('custom3')

            if journal is None:
                continue

            paper_journal_list['title'].append(entry['title'])
            paper_journal_list['journal'].append(journal)

    return pd.DataFrame(paper_journal_list)


def append_label(abstracts, journal, paper_journal):
    """
    Labels every Paper from preproc list with the relative journal classification

    :param abstracts: DataFrame with every paper data
    :type abstracts: pd.DataFrame
    :param journal: DataFrame with journal data and classification
    :param paper_journal: DataFrame with title-journal correspondence for every paper
    :type paper_journal: pd.DataFrame
    :return: abstract DataFrame with an appended column called status
    :rtype: pd.DataFrame
    """
    good = journal[journal['label'].isin(['relevant', 'keyword'])]['term']
    pgood = paper_journal['journal'].apply(lambda t: any(good.isin([t])))
    title_good = paper_journal[pgood]['title']
    cond = abstracts['title'].apply(lambda t: any(title_good.isin([t])))
    abstracts['status'] = ''
    abstracts.loc[cond, 'status'] = 'good'
    abstracts.loc[~cond, 'status'] = 'rejected'

    return abstracts


def paper_labeler(args):
    ris_path = args.ris_file
    abstracts_path = args.abstract_file
    journal_path = args.journal_file

    paper_journal = ris_reader(ris_path)
    preproc = pd.read_csv(abstracts_path, delimiter='\t', encoding='utf-8')
    journals = pd.read_csv(journal_path, delimiter='\t', encoding='utf-8')
    out = append_label(preproc, journals, paper_journal)
    out.to_csv(abstracts_path, sep='\t', index=False)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    paper_labeler(args)


if __name__ == '__main__':
    main()
