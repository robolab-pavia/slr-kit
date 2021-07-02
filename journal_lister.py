import argparse
import pathlib
import csv

from RISparser import readris


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('ris_file', type=str, help='path to the ris file')
    parser.add_argument('outfile', type=str, help='path to csv output file')

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
    journal_no_dup = list(set(journal_list))

    counter = len(journal_no_dup)
    journal_fawoc_list = [['id', 'term', 'label', 'count']]
    for i in range(counter):
        paper_count = journal_list.count(journal_no_dup[i])
        line = [str(i), journal_no_dup[i], '', paper_count]
        journal_fawoc_list.append(line)

    with open(csv_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_ALL, )
        for line in journal_fawoc_list:
            wr.writerow(line)


def main():

    cwd = pathlib.Path.cwd()

    parser = init_argparser()
    args = parser.parse_args()
    ris_path = cwd / args.ris_file
    csv_path = cwd / args.outfile

    journal_list = ris_reader(ris_path)

    journal2csv(journal_list, csv_path)


if __name__ == "__main__":
    main()
