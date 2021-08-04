import pathlib
import csv

from RISparser import readris
from slrkit_utils.argument_parser import ArgParse


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

    return [str(out)]


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    parser = ArgParse()
    parser.add_argument('ris_file', type=str, help='path to the ris file',
                        suggest_suffix='.ris')
    parser.add_argument('outfile', type=str, help='path to csv output file',
                        output=True, suggest_suffix='_journals.csv')
    return parser


def ris_reader(ris_path):
    """
    Creates a list with every Journal from the ris file

    :param ris_path: Path to the ris file
    :type ris_path: Path
    :return: List of journals
    :rtype: list
    """
    journal_list = []
    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            try:
                value = entry['secondary_title']
            except KeyError:
                value = entry.get('custom3')

            journal_list.append(value)
    return journal_list


def journal2csv(journal_list, csv_path):
    """
    Creates a csv file from the list of journals in a format that is classifiable by Fawoc

    :param journal_list: List of journals
    :type journal_list: list
    :param csv_path: Path to csv output file
    :type csv_path: Path
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
    ris_path = pathlib.Path(args.ris_file)
    csv_path = pathlib.Path(args.outfile)

    journal_list = ris_reader(ris_path)

    journal2csv(journal_list, csv_path)


if __name__ == "__main__":
    main()
