import pathlib
import csv
import sys
import pandas as pd

from slrkit_utils.argument_parser import ArgParse
from utils import assert_column


def to_record(config):
    """
    Returns the list of files to record with git
    :param config: content of the script config file
    :type config: dict[str, Any]
    :return: the list of files
    :rtype: list[str]
    :raise ValueError: if the config file does not contains the right values
    """
    out = config['outfile']
    if out is None or out == '':
        raise ValueError("'outfile' is not specified")

    fawoc_data = '_'.join([pathlib.Path(out).stem, 'fawoc_data.tsv'])
    return [str(out), fawoc_data]


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    parser = ArgParse()
    parser.add_argument('abstract_file', type=str,
                        help='path to the abstract file with papers data',
                        input=True)
    parser.add_argument('outfile', type=str, help='path to csv output file',
                        output=True, suggest_suffix='_journals.csv')
    return parser


def abstract_reader(abstract_path):
    """
    Creates a list with every Journal from the abstract file

    :param abstract_path: Path to the abstract file
    :type abstract_path: pathlib.Path
    :return: List of journals
    :rtype: list
    """

    try:
        abstr_df = pd.read_table(abstract_path, sep='\t')
    except FileNotFoundError:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(str(abstract_path)))
    assert_column(str(abstract_path), abstr_df, ['journal'])
    journal_list = list(abstr_df['journal'])

    return journal_list


def journal2csv(journal_list, csv_path):
    """
    Creates a csv file from the list of journals in a format that is classifiable by Fawoc

    :param journal_list: List of journals
    :type journal_list: list
    :param csv_path: Path to csv output file
    :type csv_path: pathlib.Path
    """
    journal_no_dup = set(journal_list)
    fieldnames = ['id', 'term', 'label', 'count']
    journal_fawoc_list = []
    for journal in journal_no_dup:
        paper_count = journal_list.count(journal)
        journal_fawoc_list.append({
            'term': journal,
            'label': '',
            'count': paper_count,
        })

    journal_fawoc_list.sort(key=lambda e: e['count'], reverse=True)
    with open(csv_path, 'w', newline='') as myfile:
        wr = csv.DictWriter(myfile, fieldnames=fieldnames, delimiter='\t',
                            quoting=csv.QUOTE_ALL, )
        wr.writeheader()
        for i, line in enumerate(journal_fawoc_list):
            line['id'] = i
            wr.writerow(line)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    journal_lister(args)


def journal_lister(args):
    abstract_path = pathlib.Path(args.abstract_file)
    csv_path = pathlib.Path(args.outfile)

    journal_list = abstract_reader(abstract_path)

    journal2csv(journal_list, csv_path)


if __name__ == "__main__":
    main()
