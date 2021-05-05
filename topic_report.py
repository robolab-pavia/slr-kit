import argparse
import json
from RISparser import readris
import collections


class Paper:
    year: int
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
    topics: list[str]


def init_argparser():
    parser = argparse.ArgumentParser()

    return parser


def prepare_papers(ris_path, json_path):
    papers_list = []

    with open(json_path) as file:
        papers_data = json.load(file)

    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            papers_list.append(entry)

    for paper in papers_list:
        for paper_data in papers_data:
            if paper["title"] == paper_data["title"]:
                topics = paper_data["topics"]
                paper["topics"] = topics
                break

    return papers_list


def report_year(papers_list):

    topics_list = []
    for paper in papers_list:
        for key in paper["topics"]:
            if int(key) not in topics_list:
                topics_list.append(int(key))

    topics_list.sort()
    topics_dict = collections.defaultdict(dict)

    for topic_id in topics_list:
        for paper in papers_list:
            topics = paper["topics"]
            if str(topic_id) in topics:
                year = int(paper["year"])
                topics_dict[topic_id][year] = topics_dict[topic_id].get(year, 0) + float(topics[str(topic_id)])

    print(topics_dict)


def main():
    # parser = init_argparser()
    # args = parser.parse_args()
    ris_path = "dsm-facchinetti-main/dsm.ris"
    json_path = "dsm_dataset/lda_docs-topics_2021-04-30_180756.json"

    papers_list = prepare_papers(ris_path, json_path)
    report_year(papers_list)

if __name__ == "__main__":
    main()