import argparse
import json
from RISparser import readris
import collections
from tabulate import tabulate
from matplotlib import pyplot as plt


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

    topics_list = []
    for paper in papers_list:
        for key in paper["topics"]:
            if int(key) not in topics_list:
                topics_list.append(int(key))

    topics_list.sort()

    return papers_list, topics_list


def report_year(papers_list, topics_list):

    topics_dict = collections.defaultdict(dict)

    for topic_id in topics_list:
        for paper in papers_list:
            topics = paper["topics"]
            if str(topic_id) in topics:
                year = int(paper["year"])
                topics_dict[topic_id][year] = topics_dict[topic_id].get(year, 0) + float(topics[str(topic_id)])

    return topics_dict


def prepare_journals(papers_list):

    journals = []
    journals_dict = dict()
    for paper in papers_list:
        if "secondary_title" in paper:
            if paper["secondary_title"] not in journals:
                journals.append(paper["secondary_title"])

    for journal in journals:
        for paper in papers_list:
            if "secondary_title" in paper:
                if paper["secondary_title"] == journal:
                    journals_dict[journal] = journals_dict.get(journal, 0) + 1

    journals_dict = sorted(journals_dict.items(), key=lambda x: x[1], reverse=True)

    return journals_dict


def report_journal_topics(journals_dict, papers_list):

    journal_topic = collections.defaultdict(dict)
    for journal, value in journals_dict[0:10]:
        for paper in papers_list:
            if "secondary_title" in paper:
                if paper["secondary_title"] == journal:
                    topics = paper["topics"]
                    for topic in topics:
                        journal_topic[journal][topic] = journal_topic[journal].get(topic, 0) + 1
    print(journal_topic)

    return journal_topic


def report_journal_years(papers_list, journals_dict):
    journal_year = collections.defaultdict(dict)

    for journal, value in journals_dict[0:10]:
        for paper in papers_list:
            if "secondary_title" in paper:
                if paper["secondary_title"] == journal:
                    year = int(paper["year"])
                    journal_year[journal][year] = journal_year[journal].get(year, 0) + 1

    return journal_year


def plot_years(topics_dict):

    for topic in topics_dict:
        sorted_dic = sorted(topics_dict[topic].items())
        x, y = zip(*sorted_dic)

        plt.grid(True)
        plt.plot(x, y, label="topic " + str(topic))

    plt.legend()
    plt.title("topics yearly graph ")
    plt.xlabel("Year")
    plt.ylabel("# of papers")
    plt.tight_layout()
    plt.show()


def prepare_tables(topics_dict):

    min_year = 2005
    first_line = ["Topic"]
    first_line.extend(list(range(min_year, 2021)))
    topic_year_list = [first_line]

    for topic in topics_dict:
        sorted_dic = sorted(topics_dict[topic].items())
        line = [topic]
        x, y = zip(*sorted_dic)
        for year in first_line[1:]:
            if year in x:
                line.append(y[x.index(year)])
            else:
                line.append(0)
        topic_year_list.append(line)

    topic_year_table = tabulate(topic_year_list, headers="firstrow", floatfmt=".3f")



def main():
    # parser = init_argparser()
    # args = parser.parse_args()
    ris_path = "dsm-facchinetti-main/dsm.ris"
    json_path = "dsm-output/lda_docs-topics_2021-05-10_151431.json"

    papers_list, topics_list = prepare_papers(ris_path, json_path)
    topics_dict = report_year(papers_list, topics_list)


if __name__ == "__main__":
    main()
