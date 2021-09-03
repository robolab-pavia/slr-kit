import pathlib
import sys

import pandas as pd
from slrkit_utils.argument_parser import ArgParse
from utils import assert_column


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    parser = ArgParse()

    parser.add_argument('bib_file', type=str, help='path to the bib file',
                        suggest_suffix='.csv')
    parser.add_argument('abstract_file', type=str,
                        help='path to the file with the abstracts of the papers',
                        input=True)
    parser.add_argument('journal_file', type=str,
                        help='path to the file with the classified journals',
                        input=True)
    parser.add_argument('--title-column', '-t', action='store', type=str,
                        default='title', dest='title_column',
                        help='Column in abstract_file with the papers titles. '
                             'If omitted %(default)r is used.')
    return parser


def bib_reader(bib_path):
    """
    Creates a list of journals and papers titles from the bib file

    :param bib_path: path to the bib file
    :type bib_path: pathlib.Path
    :return: List of titles and relative journal
    :rtype: pd.DataFrame
    """

    try:
        full_bib_df = pd.read_table(bib_path, sep='\t')
        print(full_bib_df)
        bib_df = pd.DataFrame(columns=['title', 'journal'])
        bib_df[['title', 'journal']] = full_bib_df[['title', 'journal']]
    except FileNotFoundError:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(str(bib_path)))

    return bib_df


def set_paper_status(abstracts, title_column, journal, paper_journal):
    """
    Labels every Paper from preproc list with the relative journal classification

    :param abstracts: DataFrame with every paper data
    :type abstracts: pd.DataFrame
    :param title_column: name of the column with the papers titles
    :type title_column: str
    :param journal: DataFrame with journal data and classification
    :param paper_journal: DataFrame with title-journal correspondence for every paper
    :type paper_journal: pd.DataFrame
    :return: abstract DataFrame with an appended column called status
    :rtype: pd.DataFrame
    """
    good = journal[journal['label'].isin(['relevant', 'keyword'])]['term']
    pgood = paper_journal['journal'].apply(lambda t: any(good.isin([t])))
    title_good = paper_journal[pgood][title_column]
    cond = abstracts[title_column].apply(lambda t: any(title_good.isin([t])))
    abstracts['status'] = ''
    abstracts.loc[cond, 'status'] = 'good'
    abstracts.loc[~cond, 'status'] = 'rejected'

    return abstracts


def filter_paper(args):
    bib_path = args.bib_file
    abstracts_path = args.abstract_file
    journal_path = args.journal_file

    paper_journal = bib_reader(bib_path)
    try:
        preproc = pd.read_csv(abstracts_path, delimiter='\t', encoding='utf-8')
        journals = pd.read_csv(journal_path, delimiter='\t', encoding='utf-8')
    except FileNotFoundError as err:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(err.filename))

    assert_column(str(abstracts_path), preproc, args.title_column)
    assert_column(str(journal_path), journals, ['id', 'term', 'label'])
    out = set_paper_status(preproc, args.title_column, journals, paper_journal)
    out.to_csv(abstracts_path, sep='\t', index=False)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    filter_paper(args)


if __name__ == '__main__':
    main()
