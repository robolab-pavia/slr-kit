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

    parser.add_argument('ris_path', type=str, help='path to the ris file')
    parser.add_argument('csv_path', type=str, help='path to csv output file')

    return parser


def ris_reader(ris_path):

    journal_list = []

    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            if 'alternate_title1' not in entry or 'title' not in entry:
                continue
            journal_list.append(entry['alternate_title1'])

    journal_list = list(set(journal_list))

    return journal_list


def journal2csv(journal_list, csv_path):

    counter = len(journal_list)
    journal_fawoc_list = [['id', 'term', 'label']]
    for i in range(counter):
        line = [str(i), journal_list[i], '']
        journal_fawoc_list.append(line)

    with open(csv_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_ALL, )
        for line in journal_fawoc_list:
            wr.writerow(line)


def main():

    cwd = pathlib.Path.cwd()

    parser = init_argparser()
    args = parser.parse_args()
    ris_path = cwd / args.ris_path
    csv_path = cwd / args.csv_path

    journal_list = ris_reader(ris_path)

    journal2csv(journal_list, csv_path)


if __name__ == "__main__":
    main()