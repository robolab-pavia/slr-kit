import argparse
import terms
import copy
from tqdm import tqdm

def init_argparser():
    parser = argparse.ArgumentParser()
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
    strings = []
    for item in sorted_list.items:
        if relevant_list.get(item.string) is None:
            strings.append(item.string)

    sorted_list.remove(strings)
    print(len(sorted_list))
    return sorted_list

def main():
    #parser = init_argparser()
    #args = parser.parse_args()
    tsv_path = "energy_dataset/energy_terms.csv"
    terms_list, relevant_list = list_cleaner(tsv_path)
    sorted_list = list_sorter(terms_list)
    final_list = list_comparer(sorted_list, relevant_list)


if __name__ == "__main__":
    main()