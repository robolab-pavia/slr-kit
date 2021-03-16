import json
import csv
import collections
import argparse
from matplotlib import pyplot as plt


def init_argparser():
    parser = argparse.ArgumentParser()
    # arguments needed
    parser.add_argument("json_file", type=str, help="the json file with the list of the papers")
    parser.add_argument("csv_file", type=str, help="the csv file with the papers data")
    return parser


def file_reader(json_path, csv_path):
    with open(json_path) as file:
        topics_data = json.load(file)

    papers = csv.reader(open(csv_path, encoding="utf8"), delimiter="\t")
    return topics_data, papers


def dict_builder(topics, papers):
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


def plotter(topic_dic):
    plt.style.use("fivethirtyeight")
    for dic in topic_dic:
        sorted_dic = sorted(topic_dic[dic].items())
        x, y = zip(*sorted_dic)

        plt.grid(True)
        plt.plot(x, y)
        plt.title("topic number: " + dic)
        plt.xlabel("Year")
        plt.ylabel("# of papers")
        plt.tight_layout()
        plt.show()


def main():
    parser = init_argparser()
    args = parser.parse_args()

    topics_data, papers = file_reader(args.json_file, args.csv_file)
    topics_dict = dict_builder(topics_data, papers)
    plotter(topics_dict)


if __name__ == "__main__":
    main()

