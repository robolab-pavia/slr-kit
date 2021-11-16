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

    parser.add_argument('abstract_file', type=str,
                        help='path to the file with the abstracts of the papers',
                        input=True)
    parser.add_argument('journal_file', type=str,
                        help='path to the file with the classified journals',
                        input=True)
    return parser


def abstract_reader(abstract_path):
    """
    Creates a list of journals and papers titles from the abstract file

    :param abstract_path: path to the abstract file
    :type abstract_path: pathlib.Path
    :return: List of titles and relative journal
    :rtype: pd.DataFrame
    """

    try:
        abstract_df = pd.read_csv(abstract_path, sep='\t')
        assert_column(str(abstract_path), abstract_df, ['title', 'journal'])
    except FileNotFoundError:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(str(abstract_path)))

    return abstract_df


def set_paper_status(journal, paper_journal):
    """
    Labels every Paper from preproc list with the relative journal classification

    :param journal: DataFrame with journal data and classification
    :param paper_journal: DataFrame with title-journal correspondence for every paper
    :type paper_journal: pd.DataFrame
    :return: abstract DataFrame with an appended column called status
    :rtype: pd.DataFrame
    """
    good = journal[journal['label'].isin(['relevant', 'keyword'])]['term']
    cond = paper_journal['journal'].apply(lambda t: any(good.isin([t])))
    paper_journal['status'] = ''
    paper_journal.loc[cond, 'status'] = 'good'
    paper_journal.loc[~cond, 'status'] = 'rejected'

    return paper_journal


def filter_paper(args):
    abstracts_path = args.abstract_file
    journal_path = args.journal_file

    paper_journal = abstract_reader(abstracts_path)
    try:
        journals = pd.read_csv(journal_path, delimiter='\t', encoding='utf-8')
    except FileNotFoundError as err:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(err.filename))

    assert_column(str(journal_path), journals, ['id', 'term', 'label'])
    out = set_paper_status(journals, paper_journal)
    out.to_csv(abstracts_path, sep='\t', index=False)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    filter_paper(args)


if __name__ == '__main__':
    main()
