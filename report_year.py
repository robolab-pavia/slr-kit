import json
import csv
import collections
import argparse
from matplotlib import pyplot as plt
import os.path


def init_argparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    # parser for "list" command
    parser_list = subparsers.add_parser("list", help="lists the topics and the main words from the JSON file given")
    parser_list.add_argument("topics_json", type=str, help="the json file with the list of the topics")

    # parser for "plot" command
    parser_plot = subparsers.add_parser("plot", help="plots the yearly graph of the topics given")
    parser_plot.add_argument("topics_list", type=str, help="list of the topics desired e.g. 1,3,10")
    parser_plot.add_argument("papers_json", type=str, help="the json file with the list of the papers")
    parser_plot.add_argument("papers_csv", type=str, help="the csv file with the papers data")

    return parser


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("Error: The file %s does not exist!" % arg)
        return False
    else:
        return True


def is_valid_list(plot_list, topics_dict):
    if plot_list == "all":
        return True
    else:
        split_list = plot_list.split(",")
        max_id = len(topics_dict)
        try:
            [int(x.strip()) for x in split_list]
        except:
            print("Error: Please insert a valid list of topics ids separated by a comma, e.g. 1,3,10 or 'all' ")
        else:
            if all(int(i) < max_id for i in split_list):
                return True
            else:
                print("Error: The maximum topic id is: ", max_id-1)


def file_reader(json_path, csv_path):
    with open(json_path) as file:
        topics_data = json.load(file)

    papers = csv.reader(open(csv_path, encoding="utf8"), delimiter="\t")
    return topics_data, papers


def dict_builder(topics, papers):
    # creating the dictionary with all the data needed
    papers_data = list(papers)
    papers_dic = dict()
    topic_dic = collections.defaultdict(dict)

    for paper in papers_data[1:]:
        key = int(paper[0])
        data = int(paper[2])
        papers_dic[key] = data

    for entry in topics:
        paper_id = entry["id"]
        year = papers_dic.get(paper_id)
        for topic in entry["topics"]:
            topic_dic[topic][year] = topic_dic[topic].get(year, 0) + 1

    return topic_dic


def plotter(topic_dic, topics_list):
    plt.style.use("seaborn-dark")

    if topics_list != "all":
        chosen_id = topics_list.split(",")
        for dic in topic_dic:
            if dic in chosen_id:
                sorted_dic = sorted(topic_dic[dic].items())
                x, y = zip(*sorted_dic)

                plt.grid(True)
                plt.plot(x, y, label="topic "+dic)
    else:
        for dic in topic_dic:
            sorted_dic = sorted(topic_dic[dic].items())
            x, y = zip(*sorted_dic)

            plt.grid(True)
            plt.plot(x, y, label="topic " + dic)

    plt.legend()
    plt.title("topics yearly graph ")
    plt.xlabel("Year")
    plt.ylabel("# of papers")
    plt.tight_layout()
    plt.show()


def topic_lister(topics_json):
    # function that handles the "list" command
    with open(topics_json) as file:
        topics_data = json.load(file)

    out = ""
    counter = 0

    for topic in topics_data:
        out += str(counter) + ") Name: " + topics_data.get(topic).get("name") + " - Main words: "
        for word in list(topics_data.get(topic).get("terms_probability"))[:5]:
            out += " " + word
        print(out)
        counter += 1
        out = ""


def main():
    parser = init_argparser()
    args = parser.parse_args()

    if "topics_list" in args:
        if is_valid_file(parser, args.papers_json) and is_valid_file(parser, args.papers_csv):
            topics_data, papers = file_reader(args.papers_json, args.papers_csv)
            topics_dict = dict_builder(topics_data, papers)
            if is_valid_list(args.topics_list, topics_dict):
                plotter(topics_dict, args.topics_list)
    elif "topics_json" in args:
        if is_valid_file(parser, args.topics_json):
            topic_lister(args.topics_json)


if __name__ == "__main__":
    main()

