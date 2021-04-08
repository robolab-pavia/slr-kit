import argparse
import csv
from operator import itemgetter


def init_argparser():
    parser = argparse.ArgumentParser()
    return parser


def list_sorter(terms_list):

    terms_list.pop(0)
    index = 0
    for term in list(terms_list):
        if term[2] != "relevant" and term[2] != "keyword":
            terms_list.remove(term)
        index += 1

    sorted_list = sorted(terms_list, key=itemgetter(4))

    return sorted_list


def main():
    #parser = init_argparser()
    #args = parser.parse_args()
    csv_path = "energy_dataset/energy_terms.csv"
    terms_list = list(csv.reader(open(csv_path, encoding="utf8"), delimiter="\t"))
    list_sorter(terms_list)


if __name__ == "__main__":
    main()