import argparse
import json
import csv
import os
import collections

from RISparser import readris
from tabulate import tabulate
from matplotlib import pyplot as plt
from jinja2 import Environment, FileSystemLoader
from itertools import islice


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('ris_file', type=str, help='the path to the ris file containing papers data')
    parser.add_argument('json_file', type=str, help='the path to the json file containing lda results')
    parser.add_argument('dataset_name', type=str, help='name of the dataset to be reported')

    return parser


def prepare_papers(ris_path, json_path):
    """
    Extracts all data for each paper in the ris file and the json file to create a
    list of dictionaries containing all grouped data.

    :param ris_path: path to the ris file containing papers data
    :type ris_path: str
    :param json_path: path to json file containing lda results
    :type json_path: str
    :return: the list of dictionaries and the list of topics
    :rtype: tuple[list, list]
    """
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
    """
    Creates a dictionary with number of papers published each year for every topic.
    Papers are weighted by coherence.

    :param papers_list: list of dictionaries with data for every paper
    :type papers_list: list
    :param topics_list: list of topics
    :type topics_list: list
    :return: dictionary with topic-year data
    :rtype: dict
    """

    topics_dict = collections.defaultdict(dict)

    for topic_id in topics_list:
        for paper in papers_list:
            topics = paper["topics"]
            if str(topic_id) in topics:
                year = int(paper["year"])
                topics_dict[topic_id][year] = topics_dict[topic_id].get(year, 0) + float(topics[str(topic_id)])

    return topics_dict


def prepare_journals(papers_list):
    """
    Check how many papers were published by each journal.

    :param papers_list: list with data of every paper
    :type papers_list: list
    :return: list of journals and their number of publications
    :rtype: list
    """

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
    """
    Creates a dictionary with for each journal the number of papers published for
    each topic.

    :param journals_dict: list of dictionary with data of journals and their publications
    :type journals_dict: list
    :param papers_list: list with data of every paper
    :type papers_list: list
    :return: dictionary with journal-topic data
    :rtype: dict
    """

    journal_topic = collections.defaultdict(dict)
    for journal, value in journals_dict[0:10]:
        for paper in papers_list:
            if "secondary_title" in paper:
                if paper["secondary_title"] == journal:
                    topics = paper["topics"]
                    for topic in topics:
                        journal_topic[journal][topic] = journal_topic[journal].get(topic, 0) + 1

    return journal_topic


def report_journal_years(papers_list, journals_dict):
    """
    Creates a dictionary with for each journal the number of papers published for
    each year.

    :param papers_list: list with data of every paper
    :type papers_list: list
    :param journals_dict: list of dictionary with data of journals and their publications
    :type journals_dict: list
    :return: dictionary with journal-year data
    :rtype: dict
    """
    journal_year = collections.defaultdict(dict)

    for journal, value in journals_dict[0:10]:
        for paper in papers_list:
            if "secondary_title" in paper:
                if paper["secondary_title"] == journal:
                    year = int(paper["year"])
                    journal_year[journal][year] = journal_year[journal].get(year, 0) + 1

    return journal_year


def plot_years(topics_dict, dataset_name):
    """
    Creates a plot for the number of papers published each year for each topic

    :param topics_dict: dictionary with topic-year data
    :type topics_dict: dict
    :param dataset_name: name of the dataset
    :type dataset_name: str
    """

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    for topic in islice(topics_dict, 10):
        sorted_dic = sorted(topics_dict[topic].items())
        x, y = zip(*sorted_dic)

        ax[0].plot(x, y, label="topic " + str(topic))
        ax[0].grid(True)

    for topic in islice(topics_dict, 10, None):
        sorted_dic = sorted(topics_dict[topic].items())
        x, y = zip(*sorted_dic)

        ax[1].plot(x, y, label="topic " + str(topic))
        ax[1].grid(True)

    ax[0].set_title('topics yearly graph (1st half)')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('# of papers (weighted by coherence)')
    ax[0].legend()

    ax[1].set_xlabel('year')
    ax[1].set_ylabel('# of papers (weighted by coherence)')
    ax[1].set_title('topics yearly graph (2nd half)')
    ax[1].legend()

    fig.tight_layout()
    plt.savefig(dataset_name+'_year.png')


def prepare_tables(topics_dict, journals_topic, journals_year, dataset_name):
    """
    Creates tables for every data created in previous function, with the Tabulate
    module and saves them in a directory

    :param topics_dict: dictionary with topic-year data
    :type topics_dict: dict
    :param journals_topic: dictionary with journal-topic data
    :type journals_topic: dict
    :param journals_year: dictionary with journal-year data
    :type journals_year: dict
    :param dataset_name: name of the dataset
    :type dataset_name: str
    :return: three tables for every dictionary
    :rtype: tuple[str, str, str]
    """

    min_year = 2007
    first_line = ["Topic"]
    first_line.extend(list(range(min_year, 2021)))
    topic_year_list = [first_line]

    for topic in topics_dict:
        sorted_dic = sorted(topics_dict[topic].items())
        line = [topic]
        x, y = zip(*sorted_dic)
        for year in first_line[1:]:
            if year in x:
                line.append("{:.2f}".format(y[x.index(year)]))
            else:
                line.append(0)
        topic_year_list.append(line)

    topic_year_table = tabulate(topic_year_list, headers="firstrow", tablefmt="github")

    first_line = ["Journal"]
    topics_list = list(range(0, len(topics_dict)))
    first_line.extend(topics_list)
    journal_topic_list = [first_line]

    for journal in journals_topic:
        line = [journal]
        sorted_dic = sorted(journals_topic[journal].items())
        x, y = zip(*sorted_dic)
        for topic in first_line[1:]:
            if str(topic) in x:
                line.append(y[x.index(str(topic))])
            else:
                line.append(0)
        journal_topic_list.append(line)

    journal_topic_table = tabulate(journal_topic_list, headers="firstrow", floatfmt=".3f", tablefmt="github")

    first_line = ["Journal"]
    first_line.extend(list(range(min_year, 2021)))
    journal_year_list = [first_line]
    for journal in journals_year:
        sorted_dic = sorted(journals_year[journal].items())
        line = [journal]
        x, y = zip(*sorted_dic)
        for year in first_line[1:]:
            if year in x:
                line.append(y[x.index(year)])
            else:
                line.append(0)
        journal_year_list.append(line)

    journal_year_table = tabulate(journal_year_list, headers="firstrow", floatfmt=".3f", tablefmt="github")
    if (dataset_name+'-tables') not in os.listdir():
        os.mkdir(dataset_name+'-tables')

    with open(dataset_name+'-tables/topic_year.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in topic_year_list:
            writer.writerow(row)

    with open(dataset_name+'-tables/journal_topic.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in journal_topic_list:
            writer.writerow(row)

    with open(dataset_name+'-tables/journal_year.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in journal_year_list:
            writer.writerow(row)

    return topic_year_table, journal_topic_table, journal_year_table


def main():
    parser = init_argparser()
    args = parser.parse_args()
    ris_path = args.ris_file
    json_path = args.json_file
    dataset = args.dataset_name

    papers_list, topics_list = prepare_papers(ris_path, json_path)
    topics_dict = report_year(papers_list, topics_list)
    plot_years(topics_dict, dataset)
    journals_dict = prepare_journals(papers_list)
    journals_year = report_journal_years(papers_list, journals_dict)
    journals_topics = report_journal_topics(journals_dict, papers_list)
    topic_year_table, journal_topic_table, journal_year_table = prepare_tables(topics_dict,
                                                                               journals_topics,
                                                                               journals_year,
                                                                               dataset)

    env = Environment(loader=FileSystemLoader('.'),
                      autoescape=True)

    template = env.get_template('report_markdown.md')
    year_report = dataset + '_year.png'
    md_file = template.render(year_report=year_report,
                              year_table=topic_year_table,
                              journal_topic_table=journal_topic_table,
                              journal_year_table=journal_year_table)

    with open("md_report.md", "w") as fh:
        fh.write(md_file)


if __name__ == "__main__":
    main()
