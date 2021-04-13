import argparse
import terms
import copy
from tqdm import tqdm


def init_argparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # parser for "sort" command
    parser_sort = subparsers.add_parser("sort", help="create a sorted list of only relatives and keywords terms from a"
                                                     " classified-terms tsv file")
    parser_sort.add_argument("terms_tsv", type=str, help="the tsv file with the unsorted and complete list of terms")
    parser_sort.add_argument("output", type=str, help="the path of the output file")

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


def main():
    parser = init_argparser()
    args = parser.parse_args()

    if args.command == "sort":
        tsv_path = args.terms_tsv
        terms_list, relevant_list = list_cleaner(tsv_path)
        sorted_list = list_sorter(terms_list)
        final_list = list_comparer(sorted_list, relevant_list)
        final_list.to_tsv(args.output)


if __name__ == "__main__":
    main()
