import argparse
import terms
import copy

def init_argparser():
    parser = argparse.ArgumentParser()
    return parser


def list_sorter(tsv_path):
    sorter = terms.TermList()
    sorter.from_tsv(tsv_path)
    relevant_list = terms.TermList([])
    print(len(sorter.items))
    for item in list(sorter.items):
        if item.order < 0:
            sorter.items.remove(item)
        else:
            if item.label == terms.Label.RELEVANT or item.label == terms.Label.KEYWORD:
                copy_item = copy.deepcopy(item)
                relevant_list.items.append(copy_item)
            sorter.get(item.string).order = -1
            sorter.get(item.string).label = terms.Label.NONE
    sorted_list = terms.TermList()



def main():
    #parser = init_argparser()
    #args = parser.parse_args()
    tsv_path = "energy_dataset/energy_terms.csv"
    list_sorter(tsv_path)


if __name__ == "__main__":
    main()