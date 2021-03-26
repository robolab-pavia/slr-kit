import json
import csv
import argparse
from operator import itemgetter
import os.path


def init_argparser():
    parser = argparse.ArgumentParser()
    # arguments needed
    parser.add_argument("first_file", type=str, help="the first json file with the list of topics")
    parser.add_argument("second_file", type=str, help="the second json file with the list of topics")

    return parser


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
        return False
    else:
        return True


def json_opener(path1, path2):
    with open(path1) as file1:
        topics_data1 = json.load(file1)

    with open(path2) as file2:
        topics_data2 = json.load(file2)

    return topics_data1, topics_data2


def topic_matcher(topic1, topic2):
    # for every word the abs difference is computed, if word can't find a match, difference is set as 1 by default
    partial_diff = 1.0
    metric = 0.0
    counter = 0

    for term1 in topic1["terms_probability"]:
        for term2 in topic2["terms_probability"]:
            if term1 == term2:
                partial_diff = abs(topic1["terms_probability"][term1] - topic2["terms_probability"][term2])
        metric += partial_diff
        counter += 1
        partial_diff = 1.0

    # compute average difference for the topic words and transformed in a percentage notation
    metric /= counter
    percentage_metric = round((1.0 - metric) * 100.0, 2)

    return percentage_metric


def csv_writer(topics_dict1, topics_dict2):
    data = []

    for topic1 in topics_dict1:
        for topic2 in topics_dict2:
            percentage_metric = topic_matcher(topics_dict1[topic1], topics_dict2[topic2])
            topic1_words = ""
            topic2_words = ""
            for word1 in list(topics_dict1.get(topic1).get("terms_probability"))[:5]:
                topic1_words += word1 + " "
            for word2 in list(topics_dict2.get(topic2).get("terms_probability"))[:5]:
                topic2_words += word2 + " "
            row = [topic1,
                   topic2,
                   topics_dict1[topic1].get("name"),
                   topics_dict2[topic2].get("name"),
                   percentage_metric, topic1_words,
                   topic2_words]
            data.append(row)

    data = sorted(data, key=itemgetter(4), reverse=True)

    with open("matching.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["id A",
                         "id B",
                         "topic A name",
                         "topic B name",
                         "similarity metric",
                         "topic A top 5 words",
                         "topic B top 5 words"])
        for line in data:
            writer.writerow(line)


def main():
    parser = init_argparser()
    args = parser.parse_args()

    if is_valid_file(parser, args.first_file) and is_valid_file(parser, args.second_file):
        topics_data1, topics_data2 = json_opener(args.first_file, args.second_file)
        csv_writer(topics_data1, topics_data2)


if __name__ == "__main__":
    main()
