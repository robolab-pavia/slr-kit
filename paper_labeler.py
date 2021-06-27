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
    parser.add_argument('preproc_path', type=str, help='path to preproc file')
    parser.add_argument('journal_path', type=str, help='path to journal classified file')

    return parser


def ris_reader(ris_path):
    paper_journal_list = []

    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            if 'alternate_title1' not in entry or 'title' not in entry:
                continue
            paper_journal_list.append([entry['title'], entry['alternate_title1']])

    return paper_journal_list


def csv_reader(preproc_path, journal_path):

    with open(preproc_path, encoding="utf8", newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        preproc_list = list(reader)

    with open(journal_path, encoding="utf8", newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        journal_list = list(reader)

    preproc_list = list(filter(None, preproc_list))
    return preproc_list, journal_list


def paper_labeler(preproc_list, journal_list, paper_journal_list):
    if 'status' in preproc_list[0]:
        index = preproc_list[0].index('status')
        for row in preproc_list:
            del row[index]

    preproc_list[0].append('status')

    for paper in preproc_list[1:]:
        title = paper[1]
        journal_name = ''
        status = ''
        for item in paper_journal_list:
            if title == item[0]:
                journal_name = item[1]
        for journal in journal_list[1:]:
            if journal_name == journal[1]:
                if journal[2] == 'relevant':
                    status = 'good'
                else:
                    status = 'rejected'
        paper.append(status)

    return preproc_list


def main():

    cwd = pathlib.Path.cwd()

    parser = init_argparser()
    args = parser.parse_args()

    ris_path = cwd / args.ris_path
    preproc_path = cwd / args.preproc_path
    journal_path = cwd / args.journal_path

    paper_journal_list = ris_reader(ris_path)
    preproc_list, journal_list = csv_reader(preproc_path, journal_path)
    out_list = paper_labeler(preproc_list, journal_list, paper_journal_list)

    with open(preproc_path, 'w', encoding="utf8", newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_ALL, )
        for line in out_list:
            wr.writerow(line)

if __name__ == '__main__':
    main()
