import argparse
import terms
import copy
import os
import csv
import json
from tqdm import tqdm
from pathlib import Path

import lda
import topic_matcher


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # parser for "sort" command
    parser_sort = subparsers.add_parser("sort", help="create a sorted list of only relatives and keywords terms from a"
                                                     " classified-terms tsv file")
    parser_sort.add_argument("terms_tsv", type=str, help="the tsv file with the unsorted and complete list of terms")
    parser_sort.add_argument("output", type=str, help="the path of the output file")

    # parser for "lda" command
    parser_lda = subparsers.add_parser("lda", help="runs LDA on a set of list of increasing size")
    parser_lda.add_argument("ordered_list", type=str, help="the tsv file with the ordered list of relevant terms")
    parser_lda.add_argument("preproc_file", type=str, help="the tsv file with data from all papers")
    parser_lda.add_argument("output_path", type=str, help="path to the output folder desired")
    parser_lda.add_argument("--min", type=int, default=1000)
    parser_lda.add_argument("--increment", type=int, default=200)

    # parser for "match" command
    parser_match = subparsers.add_parser("match", help="matches and compares the results from the LDA subcommand")
    parser_match.add_argument("output", type=str, help="path to the directory containing lda directory")

    return parser


def list_cleaner(tsv_path):
    """
    Cleans the list of terms keeping only relevant and keyword classified terms.

    :param tsv_path: path to the csv file with the complete list of terms.
    :type tsv_path: str
    :return: the original list and the relevant-keyword only list.
    :rtype: tuple[TermsList, TermsList]
    """
    terms_list = terms.TermList()
    terms_list.from_tsv(tsv_path)
    relevant_list = terms.TermList([])

    print("Starting list cleaning...")

    for item in tqdm(list(terms_list.items)):

        if item.label == terms.Label.RELEVANT or item.label == terms.Label.KEYWORD:
            copy_item = copy.deepcopy(item)
            relevant_list.items.append(copy_item)
        terms_list.get(item.string).order = -1
        terms_list.get(item.string).label = terms.Label.NONE

    print("List has been cleaned")

    return terms_list, relevant_list


def list_sorter(terms_list):
    """
    Sorts the list according to Fawoc submitting order.

    :param terms_list: the original list of terms.
    :type terms_list: TermsList
    :return: the sorted list of all terms.
    :rtype: TermsList
    """
    multigram_index = 0
    print("Starting list sorting...")

    for term in terms_list.items:
        if len(term.string.split()) > 1:
            break
        multigram_index = term.index
    sorted_list = terms.TermList([])

    for item in tqdm(terms_list.items[:multigram_index]):
        next_sublist, nc = terms_list.return_related_items(item.string)
        strings = next_sublist.get_strings()
        for string in strings:
            terms_list.get(string).order = 1
        sorted_list = sorted_list + next_sublist

    return sorted_list


def list_comparer(sorted_list, relevant_list):
    """
    Removes from the sorted list every term that is not either classified as relevant or keyword.

    :param sorted_list: the sorted list containing all terms.
    :param relevant_list: the list containing only relevant and keyword terms.
    :return: the sorted list of only relevant and keyword classified terms.
    :rtype: TermsList
    """
    index = 0

    for item in list(sorted_list.items):
        if relevant_list.get(item.string) is None:
            sorted_list.items.remove(item)
        else:
            sorted_list.get(item.string).label = relevant_list.get(item.string).label
            sorted_list.get(item.string).index = index
            index += 1

    return sorted_list


def lda_iterator(terms_file, preproc_file, output_path, minimum, increment):
    """
    Executes LDA algorithm on an increasing size sorted list of only relevant or keyword classified terms.

    :param terms_file: the path to the ordered list of terms.
    :type terms_file: str
    :param preproc_file: the path to the preproc_file.csv containing data of every paper.
    :type preproc_file: str
    :param output_path: path to the directory where the results will be saved.
    :type output_path: str
    :param minimum: minimum number of terms to consider from the sorted list.
    :type minimum: int
    :param increment: the value of words to consider in addiction to the previous iteration.
    :type increment: int
    """

    dict_path = output_path + "/dictionary"
    list_path = output_path + "/list"
    lda_path = output_path + "/lda"

    Path(dict_path).mkdir(parents=True, exist_ok=True)
    Path(list_path).mkdir(parents=True, exist_ok=True)
    Path(lda_path).mkdir(parents=True, exist_ok=True)

    x = minimum
    y = increment
    terms_list = list(csv.reader(open(terms_file, encoding="utf8"), delimiter="\t"))
    max_len = len(terms_list)

    # writing the increasing size list where x is the beginning value and y is the increment
    while x <= max_len:
        with open(list_path + "/list_" + str(x) + ".csv", "w", newline="", encoding='utf-8') as temp_out:
            writer = csv.writer(temp_out, delimiter="\t")
            writer.writerow(terms_list[0])
            for row in terms_list[1:x+1]:
                writer.writerow(row)
        x += y

    # applying LDA algorithm to every list generated from upper loop and saving results in json file
    for filename in tqdm(os.listdir(list_path)):
        docs, titles = lda.prepare_documents(preproc_file, list_path + "/" + filename, True, ('keyword', 'relevant'))
        model, dictionary = lda.train_lda_model(docs, 20, "auto", "auto", 1, 1)
        dictionary.save_as_text(dict_path + "/dic_" + filename[:-4] + ".txt")
        docs_topics, topics, avg_topic_coherence = lda.prepare_topics(model, docs, titles, dictionary)
        topic_file = lda_path + "/topics_" + filename[:-4] + ".json"
        with open(topic_file, 'w') as file:
            json.dump(docs_topics, file, indent='\t')


def matcher(output_path):
    """
    Matches the various execution of LDA from lda_iterator.

    :param output_path: path to the directory containing lda subcommand results.
    :type output_path: str
    """
    Path(output_path+"/matches").mkdir(parents=True, exist_ok=True)

    temp_dir = output_path+"/matches"
    lda_dir = output_path+"/lda"

    file_counter = 0
    file_list = sorted(os.listdir(lda_dir), key=len)

    # filtering only topic_list_ files
    for filename in os.listdir(lda_dir):
        if filename[:12] == "topics_list_":
            file_counter += 1
        else:
            file_list.remove(filename)

    # matching every consecutive pair of files
    for x in range(file_counter-1):
        name = file_list[x][-9:-5]+"-"+file_list[x+1][-9:-5]
        name = name.replace("_", "")
        topics_data1, topics_data2 = topic_matcher.json_opener(lda_dir+"/"+file_list[x],
                                                               lda_dir+"/"+file_list[x+1])
        topic_matcher.csv_writer(topics_data1, topics_data2, temp_dir + "/" + name + ".csv")

    # computing the mean for the top 20 topic pairs from each topic_matcher output file
    with open(output_path+"/overall_result.csv", "w", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["Terms A",
                         "Terms B",
                         "Avg top 20 topics",
                         "Actual terms A",
                         "Actual terms B"])

        # writing results in the overall_results.csv file and saving the dictionary of actually used terms
        for filename in sorted(os.listdir(temp_dir), key=len):
            avg = 0
            match_list = list(csv.reader(open(temp_dir+"/"+filename, encoding="utf8"), delimiter=","))
            line = filename[:-4].split("-")
            lines_a = sum(1 for line in open(output_path + '/dictionary/dic_list_'+line[0]+".txt")) - 1
            lines_b = sum(1 for line in open(output_path + '/dictionary/dic_list_'+line[1]+".txt")) - 1
            for row in match_list[1:21]:
                avg += float(row[4])

            avg /= 20
            avg = round(avg, 2)
            line.append(str(avg))
            line.append(str(lines_a))
            line.append(str(lines_b))
            writer.writerow(line)


def main():
    parser = init_argparser()
    args = parser.parse_args()

    if args.command == "sort":
        tsv_path = args.terms_tsv
        terms_list, relevant_list = list_cleaner(tsv_path)
        sorted_list = list_sorter(terms_list)
        final_list = list_comparer(sorted_list, relevant_list)
        final_list.to_tsv(args.output)
        print("List has been sorted")
    elif args.command == "lda":
        lda_iterator(args.ordered_list, args.preproc_file, args.output_path, args.min, args.increment)
    elif args.command == "match":
        matcher(args.output)


if __name__ == "__main__":
    main()
