import argparse
import terms
import copy
from tqdm import tqdm
from pathlib import Path
import csv
import lda
import topic_matcher
import os
import json


def init_argparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # parser for "sort" command
    parser_sort = subparsers.add_parser("sort", help="create a sorted list of only relatives and keywords terms from a"
                                                     " classified-terms tsv file")
    parser_sort.add_argument("terms_tsv", type=str, help="the tsv file with the unsorted and complete list of terms")
    parser_sort.add_argument("output", type=str, help="the path of the output file")

    #parser for "lda" command
    parser_lda = subparsers.add_parser("lda", help="todo")
    parser_lda.add_argument("ordered_list", type=str, help="the tsv file with the ordered list of relevant terms")
    parser_lda.add_argument("preproc_file", type=str, help="the tsv file with data from all papers")
    parser_lda.add_argument("output_path", type=str, help="path to the output folder desired")
    parser_lda.add_argument("--min", type=int, default=1000)
    parser_lda.add_argument("--increment", type=int, default=200)

    return parser


def list_cleaner(tsv_path):
    terms_list = terms.TermList()
    terms_list.from_tsv(tsv_path)
    relevant_list = terms.TermList([])

    print("Starting list cleaning...")

    for item in tqdm(list(terms_list.items)):
        if item.order < 0:
            terms_list.items.remove(item)
        else:
            if item.label == terms.Label.RELEVANT or item.label == terms.Label.KEYWORD:
                copy_item = copy.deepcopy(item)
                relevant_list.items.append(copy_item)
            terms_list.get(item.string).order = -1
            terms_list.get(item.string).label = terms.Label.NONE

    print("List has been cleaned")

    return terms_list, relevant_list


def list_sorter(terms_list):
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

    print("List has been sorted")

    return sorted_list


def list_comparer(sorted_list, relevant_list):

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

    Path(output_path).mkdir(parents=True, exist_ok=True)

    x = minimum
    y = increment
    terms_list = list(csv.reader(open(terms_file, encoding="utf8"), delimiter="\t"))
    max_len = len(terms_list)

    while x <= max_len:
        with open(output_path + "/list_" + str(x) + ".csv", "w", newline="", encoding='utf-8') as temp_out:
            writer = csv.writer(temp_out, delimiter="\t")
            writer.writerow(terms_list[0])
            for row in terms_list[1:x+1]:
                writer.writerow(row)
        x += y

    for filename in tqdm(os.listdir(output_path)):
        docs, titles = lda.prepare_documents(preproc_file, output_path + "/" + filename, True, ('keyword', 'relevant'))
        model, dictionary = lda.train_lda_model(docs, 20, "auto", "auto", 0.5, 20)
        docs_topics, topics, avg_topic_coherence = lda.prepare_topics(model, docs, titles, dictionary)
        os.remove(output_path + "/" + filename)
        topic_file = output_path + "/" + "topics_" + filename[:-4] + ".json"
        with open(topic_file, 'w') as file:
            json.dump(docs_topics, file, indent='\t')


def main():
    parser = init_argparser()
    args = parser.parse_args()

    if args.command == "sort":
        tsv_path = args.terms_tsv
        terms_list, relevant_list = list_cleaner(tsv_path)
        sorted_list = list_sorter(terms_list)
        final_list = list_comparer(sorted_list, relevant_list)
        final_list.to_tsv(args.output)
    elif args.command == "lda":
        lda_iterator(args.ordered_list, args.preproc_file, args.output_path, args.min, args.increment)


if __name__ == "__main__":
    main()
