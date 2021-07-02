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
    parser.add_argument('abstract_file', type=str,
                        help='path to the file with the abstracts of the papers')
    parser.add_argument('journal_file', type=str,
                        help='path to the file with the classified journals')

    return parser


def ris_reader(ris_path):
    """
    Creates a list of journals and papers titles from the ris file

    :param ris_path: path to the ris file
    :type ris_path: Path
    :return: List of titles and relative journal
    :rtype: list
    """
    paper_journal_list = []

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

            paper_journal_list.append([entry['title'], journal])

    return paper_journal_list


def csv_reader(preproc_path, journal_path):
    """
    Creates 2 lists from the preproc file and the journal file

    :param preproc_path: Path to preproc file
    :type preproc_path: Path
    :param journal_path: Path to journal file
    :type journal_path: Path
    :return: list of preproc file and journal file
    :rtype: tuple(list, list)
    """
    with open(preproc_path, encoding='utf8', newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        preproc_list = list(reader)

    with open(journal_path, encoding='utf8', newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        journal_list = list(reader)

    preproc_list = list(filter(None, preproc_list))
    return preproc_list, journal_list


def paper_labeler(abstracts, journal, paper_journal_list):
    """
    Labels every Paper from preproc list with the relative journal classification

    :param abstracts: List with every paper data
    :type abstracts: list
    :param journal: List with journal data and classification
    :type journal: list
    :param paper_journal_list: list with title-journal for every paper
    :type paper_journal_list: list
    :return: Preproc list with an appended column called status
    :rtype: list
    """
    if 'status' in abstracts[0]:
        index = abstracts[0].index('status')
        for row in abstracts:
            del row[index]

    abstracts[0].insert(1, 'status')

    for paper in abstracts[1:]:
        title = paper[1]
        journal_name = ''
        status = ''
        for item in paper_journal_list:
            if title == item[0]:
                journal_name = item[1]
        for journal in journal[1:]:
            if journal_name == journal[1]:
                if journal[2] == 'relevant':
                    status = 'good'
                else:
                    status = 'rejected'
        paper.insert(1, status)

    return abstracts


def main():
    parser = init_argparser()
    args = parser.parse_args()

    ris_path = args.ris_file
    abstracts_path = args.abstract_file
    journal_path = args.journal_file

    paper_journal_list = ris_reader(ris_path)
    abstracts_list, journal_list = csv_reader(abstracts_path, journal_path)
    out_list = paper_labeler(abstracts_list, journal_list, paper_journal_list)

    with open(abstracts_path, 'w', encoding='utf8', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_ALL, )
        for line in out_list:
            wr.writerow(line)


if __name__ == '__main__':
    main()
